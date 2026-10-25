APP      := build/VoxelReverie.app
BIN      := $(APP)/Contents/MacOS/voxelreverie
RES      := $(APP)/Contents/Resources
SOURCES  := $(shell find Sources -name '*.swift') Package.swift

all: $(BIN) $(RES)/Saver.metallib $(APP)/Contents/Info.plist

# Copied to a new file and moved over the old: the kernel kills a signed binary rewritten in place.
$(BIN): $(SOURCES)
	@mkdir -p "$(dir $@)"
	swift build -c release --product voxelreverie
	cp "$$(swift build -c release --show-bin-path)/voxelreverie" "$@.new"
	mv -f "$@.new" "$@"

$(RES)/Saver.metallib: Shaders/Saver.metal
	@mkdir -p "$(RES)"
	xcrun metal -O3 -std=metal3.1 Shaders/Saver.metal -o "$@"

$(APP)/Contents/Info.plist: Info.plist
	@mkdir -p "$(dir $@)"
	cp Info.plist "$@"

clean:
	trash build .build

.PHONY: all clean
