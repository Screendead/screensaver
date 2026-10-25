APP      := build/VoxelReverie.app
BIN      := $(APP)/Contents/MacOS/voxelreverie
RES      := $(APP)/Contents/Resources
SOURCES  := $(shell find Sources -name '*.swift') Package.swift
SHADERS  := $(shell find Shaders -name '*.metal')

all: $(BIN) $(RES)/Saver.metallib $(APP)/Contents/Info.plist

# Copied to a new file and moved over the old: the kernel kills a signed binary rewritten in place.
$(BIN): $(SOURCES)
	@mkdir -p "$(dir $@)"
	swift build -c release --product voxelreverie
	cp "$$(swift build -c release --show-bin-path)/voxelreverie" "$@.new"
	mv -f "$@.new" "$@"

$(RES)/Saver.metallib: $(SHADERS)
	@mkdir -p "$(RES)"
	xcrun metal -O3 -std=metal3.1 Shaders/Saver.metal -o "$@"

$(APP)/Contents/Info.plist: Info.plist
	@mkdir -p "$(dir $@)"
	cp Info.plist "$@"

# The unit tests run a few small kernels, so under the GPU lock; golden.sh renders through vrr, which takes it itself.
test: all
	tools/vrgpu -- swift test
	tools/golden.sh

clean:
	trash build .build

.PHONY: all test clean
