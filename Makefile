# Image build/publish. Run on a build host (hg4os), not on BlueVela.
#   make image EXTRA=swebench
#   make publish-image EXTRA=swebench
# ICR namespace of the ETE CIL12 tenant, e.g. us.icr.io/<namespace>.
REGISTRY ?=
EXTRA ?= swebench
TAG ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo dev)
IMAGE := $(REGISTRY)/sage2-evals-$(EXTRA):$(TAG)
ENGINE ?= podman

.PHONY: test image publish-image image-ref

test:
	uv run --extra $(EXTRA) pytest -q

image:
	@test -n "$(REGISTRY)" || { echo 'set REGISTRY=<icr host>/<namespace>'; exit 1; }
	$(ENGINE) build --platform linux/amd64 --build-arg EXTRA=$(EXTRA) -f docker/Dockerfile -t $(IMAGE) .

publish-image: image
	$(ENGINE) push $(IMAGE)
	@echo "published $(IMAGE)"

image-ref:
	@echo $(IMAGE)
