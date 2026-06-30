class Visualizer:
    @staticmethod
    def __enforce_tuple(unk_input):
        if isinstance(unk_input, str):
            unk_input = (unk_input,)
        unk_input = tuple(ax.lower() for ax in unk_input)
        return unk_input

    @staticmethod
    def __open_ovf(filename, grid_dims):
      nx, ny, nz = grid_dims
      with open(filename, "rb") as f:
          for line in f:
              line_str = line.decode('utf-8', errors='ignore').strip()
              if "# Begin: Data Binary" in line_str:
                  break
          _ = np.fromfile(f, dtype=np.float32, count=1)
          total_nodes = nx * ny * nz
          raw_binary = np.fromfile(f, dtype=np.float32, count=total_nodes * 3)

      raw_vectors = raw_binary.reshape((nz, ny, nx, 3))
      spatial_vectors = np.transpose(raw_vectors, (2, 1, 0, 3))
      return spatial_vectors

    @staticmethod
    def __select_plotting_axes(decay_axes, grid_dims):
        nx, ny, nz = grid_dims
        dims = {'x': nx, 'y': ny, 'z': nz}
        all_axes = ['x', 'y', 'z']

        if len(decay_axes) == 2:
            h_axis, v_axis = decay_axes[0], decay_axes[1]
        elif len(decay_axes) == 1:
            h_axis = decay_axes[0]
            remaining = [ax for ax in all_axes if ax != h_axis]
            v_axis = remaining[0] if dims[remaining[0]] >= dims[remaining[1]] else remaining[1]
        else:
            h_axis, v_axis = 'x', 'y'

        axis_pair = (h_axis, v_axis)
        flat_axis = [ax for ax in all_axes if ax != h_axis and ax != v_axis][0]

        return axis_pair, flat_axis

    @staticmethod
    def __get_slice_idx(grid_dims, flat_axis):
        nx, ny, nz = grid_dims
        midpoints = {'x': nx // 2, 'y': ny // 2, 'z': nz // 2}
        slice_idx = midpoints[flat_axis]
        return slice_idx

    @staticmethod
    def __identify_axes_plotting_info(axis_pair, slice_idx):
        axis_key = "".join(sorted(axis_pair))
        full_slice = slice(None)

        plot_config = {
            "xy": {
                "slices": (full_slice, full_slice, slice_idx),
                "h_ax": "x", "v_ax": "y",
                "h_lbl": "X Position (nm)", "v_lbl": "Y Position (nm)"
            },
            "yz": {
                "slices": (slice_idx, full_slice, full_slice),
                "h_ax": "y", "v_ax": "z",
                "h_lbl": "Y Position (nm)", "v_lbl": "Z Position (nm)"
            },
            "xz": {
                "slices": (full_slice, slice_idx, full_slice),
                "h_ax": "x", "v_ax": "z",
                "h_lbl": "X Position (nm)", "v_lbl": "Z Position (nm)"
            }
        }
        cfg = plot_config[axis_key]
        return cfg

    @staticmethod
    def __setup_plot_based_on_axes(axis_pair, grid_dims, cell_dims, slice_idx, spatial_vectors):
        nx, ny, nz = grid_dims
        dx, dy, dz = cell_dims

        coords_1d = {
            'x': (np.arange(nx) - (nx - 1) / 2) * dx * 1e9,
            'y': (np.arange(ny) - (ny - 1) / 2) * dy * 1e9,
            'z': (np.arange(nz) - (nz - 1) / 2) * dz * 1e9
        }

        cfg = Visualizer.__identify_axes_plotting_info(axis_pair, slice_idx)

        Vx = spatial_vectors[cfg["slices"] + (0,)]
        Vy = spatial_vectors[cfg["slices"] + (1,)]
        Vz = spatial_vectors[cfg["slices"] + (2,)]

        # Generate grids and labels dynamically
        Grid_H, Grid_V = np.meshgrid(coords_1d[cfg["h_ax"]], coords_1d[cfg["v_ax"]], indexing='ij')
        h_label, v_label = cfg["h_lbl"], cfg["v_lbl"]

        return Vx, Vy, Vz, Grid_H, Grid_V, h_label, v_label

    @staticmethod
    def __render_plot(Vx, Vy, Vz, V_mag, Grid_H, Grid_V, h_label, v_label, field_axis, field_var, field_units):
        fig = plt.figure(figsize=(16, 12))
        components = [Vx, Vy, Vz, V_mag]
        titles = [
            f"Field Component ${field_var}_x$ (Field Axis: {field_axis})",
            f"Field Component ${field_var}_y$ (Field Axis: {field_axis})",
            f"Field Component ${field_var}_z$ (Field Axis: {field_axis})",
            f"Total Field Magnitude $|{field_var}|$"
        ]
        field_units_wrapped = ""
        if len(field_units) != 0:
            field_units_wrapped = f"({field_units})"
        for i in range(4):
            ax = fig.add_subplot(2, 2, i+1, projection='3d')
            surf = ax.plot_surface(Grid_H, Grid_V, components[i], cmap='viridis', edgecolor='none')

            ax.set_title(titles[i], fontsize=12)
            ax.set_xlabel(h_label)
            ax.set_ylabel(v_label)
            ax.set_zlabel(f"Field {field_units_wrapped}")
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label=field_units)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def __setup_temp_image_folder():
      temp_img_folder = '/content/temp_frames'

      if os.path.exists(temp_img_folder):
        import shutil
        shutil.rmtree(temp_img_folder)
      os.makedirs(temp_img_folder, exist_ok=True)
      return temp_img_folder

    @staticmethod
    def __grab_ovf_files(folder_path, prefix_pattern):
      ovf_files = sorted(glob.glob(os.path.join(folder_path, f'{prefix_pattern}*.ovf')))
      if not ovf_files:
        print("No OVF files found. Check your file extension or path.")
        return
      print(f"Processing {len(ovf_files)} OVF files into uniform graphs...")
      return ovf_files

    @staticmethod
    def __generate_ovf_2D_graphs(ovf_files, temp_img_folder):
      import pyovf
      png_frames = []
      for i, ovf_path in enumerate(ovf_files):
        # Read the vector field data into a NumPy array
        ovf_obj = pyovf.read(ovf_path)
        data = ovf_obj.data

        # Take a 2D slice (adjust index depending on your simulation geometry)
        z_slice = data.shape[0] // 2
        mx = data[z_slice, :, :, 0] # X component

        # FIX: Explicitly set figure size
        fig, ax = plt.subplots(figsize=(6, 5))

        # FIX: vmin/vmax keeps the colorbar scale locked across the whole video
        im = ax.imshow(mx, cmap='RdBu', origin='lower', vmin=-1.0, vmax=1.0)
        ax.set_title(f"Frame {i}: Magnetization $M_x$")

        # FIX: Lock colorbar dimensions so it does not distort the frame size
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_aspect('equal')

        # FIX: Force strict margins instead of dynamic bounding boxes
        plt.tight_layout()

        # Save frame as a temporary PNG
        frame_path = os.path.join(temp_img_folder, f"frame_{i:06d}.png")
        plt.savefig(frame_path, dpi=150)
        plt.close(fig) # Prevent memory leaks in Colab

        png_frames.append(frame_path)
      return png_frames

    @staticmethod
    def __make_png_movie(png_files, output_name, fps=10):
      if not png_files:
          print("No matching PNG files found. Double-check your path.")
      else:
          print(f"Compiling video from {png_files[0]} to {png_files[-1]}...")
          clip = ImageSequenceClip(png_files, fps)

          # 4. Save video locally to the Colab session
          output_path = f'{output_name}.mp4'
          clip.write_videofile(output_path, codec='libx264', logger=None)

          # 5. Display the player directly in your browser
          mp4_data = open(output_path, 'rb').read()
          data_url = f"data:video/mp4;base64,{b64encode(mp4_data).decode()}"

          video_html = f"""
          <video width="640" controls autoplay loop>
              <source src="{data_url}" type="video/mp4">
          </video>
          """
          display(HTML(video_html))

    @staticmethod
    def get_plot_data(filename, grid_dims, cell_dims, decay_axes=('x', 'y')):
        decay_axes = Visualizer.__enforce_tuple(decay_axes)
        spatial_vectors = Visualizer.__open_ovf(filename, grid_dims)
        axis_pair, flat_axis = Visualizer.__select_plotting_axes(decay_axes, grid_dims)
        slice_idx = Visualizer.__get_slice_idx(grid_dims, flat_axis)
        Bx, By, Bz, Grid_H, Grid_V, h_label, v_label = Visualizer.__setup_plot_based_on_axes(
            axis_pair, grid_dims, cell_dims, slice_idx, spatial_vectors
        )
        B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
        return Bx, By, Bz, Grid_H, Grid_V, B_mag, h_label, v_label

    @staticmethod
    def plot_field_ovf(filename, grid_dims, cell_dims, field_var='F', field_units='', decay_axes=('x', 'y'), field_axis='z'):
        Bx, By, Bz, Grid_H, Grid_V, B_mag, h_label, v_label = Visualizer.get_plot_data(
            filename, grid_dims, cell_dims, decay_axes=decay_axes
        )

        Visualizer.__render_plot(Bx, By, Bz, B_mag, Grid_H, Grid_V, h_label, v_label, field_axis, field_var, field_units)

    @staticmethod
    def display_ovf_2D_graph_movie(folder_path, prefix_pattern, fps=10):
      temp_img_folder = Visualizer.__setup_temp_image_folder()

      ovf_files = Visualizer.__grab_ovf_files(folder_path, prefix_pattern)

      if not ovf_files:
        return

      png_frames = Visualizer.__generate_ovf_2D_graphs(ovf_files, temp_img_folder)

      if not png_frames:
        return

      print("\nStitching graphs into a video...")
      Visualizer.__make_png_movie(png_frames, 'ovf_simulation_movie', fps)

    @staticmethod
    def make_png_movie_from_folder(folder_path, output_name, pattern='m', fps=10):
      png_files = sorted(glob.glob(os.path.join(folder_path, f'{pattern}*.png')))
      Visualizer.__make_png_movie(png_files, output_name, fps=fps)

    @staticmethod
    def animate_ovf_series(
            grid_dims,
            cell_dims,
            plotting='B',
            pattern="B_ext*.ovf",
            output="gaussian_pulse.mp4",
            decay_axes=('x', 'y'),
            fps=20):
        """
        Animate a time series of OVF files using the same loading/slicing
        pipeline as get_plot_data / plot_field_ovf, so no separate
        loader (e.g. load_generalized_ovf) is needed.
        """
        files = sorted(glob.glob(pattern))

        if len(files) == 0:
            raise FileNotFoundError(f"No files matched pattern {pattern}")

        print(f"Found {len(files)} OVF files")

        # ----------------------------------------
        # Load every frame ONCE up front. The original version read each
        # file from disk twice (once to find max_field, once again inside
        # update() during animation) -- caching removes that redundant I/O
        # and redundant slicing/meshgrid work entirely.
        # ----------------------------------------
        frames_data = [
            Visualizer.get_plot_data(f, grid_dims, cell_dims, decay_axes=decay_axes)
            for f in files
        ]

        # h_label/v_label are identical across frames (grid_dims/decay_axes
        # don't change between files), so just grab them once.
        h_label, v_label = frames_data[0][6], frames_data[0][7]

        max_field = max(
            max(np.max(np.abs(Bx)), np.max(np.abs(By)), np.max(np.abs(Bz)), np.max(Bmag))
            for Bx, By, Bz, _, _, Bmag, _, _ in frames_data
        )

        fig = plt.figure(figsize=(16, 12))
        titles = [
            rf"${plotting}_x$", 
            rf"${plotting}_y$", 
            rf"${plotting}_z$", 
            rf"$|{plotting}|$"
        ]

        def update(frame):
            fig.clear()

            Bx, By, Bz, Grid_H, Grid_V, Bmag, _, _ = frames_data[frame]
            data = [Bx, By, Bz, Bmag]

            for i in range(4):
                ax = fig.add_subplot(2, 2, i + 1, projection='3d')
                ax.plot_surface(Grid_H, Grid_V, data[i], cmap='viridis', edgecolor='none')
                ax.set_zlim(0, max_field)
                ax.set_title(f"{titles[i]}   Frame {frame}")
                ax.set_xlabel(h_label)
                ax.set_ylabel(v_label)
                ax.set_zlabel("Field (T)")

            plt.tight_layout()

        ani = FuncAnimation(fig, update, frames=len(files), interval=50)
        ani.save(output, writer='ffmpeg', fps=fps)

        print(f"Saved movie: {output}")
