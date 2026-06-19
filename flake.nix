{
  description = "unshape - constructive media generation";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, fenix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        fenixPkgs = fenix.packages.${system};
        # Stable toolchain for the default devShell (>=1.92 required by eframe 0.34).
        rustToolchain = fenixPkgs.stable.withComponents [
          "cargo"
          "rustc"
          "rust-src"
          "clippy"
          "rustfmt"
        ];
        # Nightly toolchain for fuzzing
        nightlyToolchain = fenixPkgs.latest.withComponents [
          "cargo"
          "rustc"
          "rust-src"
        ];
      in
      {
        devShells.default = pkgs.mkShell rec {
          buildInputs = with pkgs; [
            stdenv.cc.cc
            # Rust toolchain (fenix stable, >=1.92)
            rustToolchain
            rust-analyzer
            # Fast linker for incremental builds
            mold
            clang
            # JS tooling: docs
            bun
            # GUI runtime libs (winit/wgpu/eframe dlopen these at runtime).
            # On NixOS these are not on the system loader path, so expose them
            # via LD_LIBRARY_PATH below or window creation fails (NoWaylandLib).
            wayland
            libxkbcommon
            libGL
            vulkan-loader
            libx11
            libxcursor
            libxrandr
            libxi
            fontconfig
            freetype
          ];
          LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath buildInputs}:$LD_LIBRARY_PATH";
          # NixOS exposes the system GPU driver + Vulkan ICDs under the stable
          # /run/opengl-driver symlink, NOT on the default loader path. The
          # vulkan-loader we ship in buildInputs otherwise enumerates zero
          # adapters, so wgpu fails with RequestAdapterError(NotFound). Point
          # the loader at the system ICD JSONs and expose the driver .so dir.
          # (Used with `nix develop --impure` so the runtime path is visible.)
          shellHook = ''
            export LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH"
            export VK_DRIVER_FILES="/run/opengl-driver/share/vulkan/icd.d"
          '';
        };

        # Fuzzing shell with nightly Rust
        devShells.fuzz = pkgs.mkShell rec {
          buildInputs = with pkgs; [
            stdenv.cc.cc
            # Nightly Rust for fuzzing
            nightlyToolchain
            # Fuzzing tool
            cargo-fuzz
            # Fast linker
            mold
            clang
          ];
          LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath buildInputs}:$LD_LIBRARY_PATH";
        };
      }
    );
}
