APP      := build/VoxelReverie.app
BIN      := $(APP)/Contents/MacOS/voxelreverie
RES      := $(APP)/Contents/Resources
SOURCES  := $(wildcard Sources/*.swift)

all: $(BIN) $(RES)/Saver.metallib $(APP)/Contents/Info.plist

$(BIN): $(SOURCES)
	@mkdir -p "$(dir $@)"
	swiftc -O -target arm64-apple-macos14 -framework AppKit -framework Metal -framework MetalKit -framework IOKit -framework AVFoundation -framework CoreMedia -framework CoreVideo -framework VideoToolbox $(SOURCES) -o "$@"

$(RES)/Saver.metallib: Shaders/Saver.metal
	@mkdir -p "$(RES)"
	xcrun metal -O3 -std=metal3.1 Shaders/Saver.metal -o "$@"

$(APP)/Contents/Info.plist: Info.plist
	@mkdir -p "$(dir $@)"
	cp Info.plist "$@"

clean:
	trash build

.PHONY: all clean
