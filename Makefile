# Image build/publish. Run on a build host (hg4os), not on BlueVela.
#   make image EXTRA=swebench
#   make publish-image EXTRA=swebench
# ICR namespace (ETE CIL12 tenant) for the sage2 images.
REGISTRY ?= us.icr.io/cil15-shared-registry
# hg4os builds as root inside a container: BUILD_FLAGS='--isolation=chroot --cgroup-manager=cgroupfs'
BUILD_FLAGS ?=
EXTRA ?= swebench
TAG ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo dev)
IMAGE := $(REGISTRY)/sage2-evals-$(EXTRA):$(TAG)
ENGINE ?= podman

.PHONY: test image publish-image image-ref

test:
	uv run --extra $(EXTRA) pytest -q

image:
	@test -n "$(REGISTRY)" || { echo 'set REGISTRY=<icr host>/<namespace>'; exit 1; }
	$(ENGINE) build $(BUILD_FLAGS) --platform linux/amd64 --build-arg EXTRA=$(EXTRA) -f docker/Dockerfile -t $(IMAGE) .

publish-image: image
	$(ENGINE) push $(IMAGE)
	@echo "published $(IMAGE)"

image-ref:
	@echo $(IMAGE)
