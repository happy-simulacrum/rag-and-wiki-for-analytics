VERSION ?= 0.1.0
BUNDLE  := bundle
QDRANT  := qdrant/qdrant:v1.15.1
WEBUI   := ghcr.io/open-webui/open-webui:v0.6.33

.PHONY: bundle test clean

test:
	.venv/bin/python -m pytest tests -q

bundle:
	docker build -f deploy/Dockerfile.backend -t codeqa-backend:$(VERSION) .
	docker pull $(QDRANT)
	docker pull $(WEBUI)
	rm -rf $(BUNDLE) codeqa-bundle-$(VERSION).tar.gz
	mkdir -p $(BUNDLE)/images
	docker save -o $(BUNDLE)/images/images.tar \
	  codeqa-backend:$(VERSION) $(QDRANT) $(WEBUI)
	cp deploy/docker-compose.yml deploy/deploy.sh deploy/update.sh README.md $(BUNDLE)/
	chmod +x $(BUNDLE)/deploy.sh $(BUNDLE)/update.sh
	tar -czf codeqa-bundle-$(VERSION).tar.gz -C $(BUNDLE) .
	@echo "==> codeqa-bundle-$(VERSION).tar.gz готов к переносу в контур"

clean:
	rm -rf $(BUNDLE) codeqa-bundle-*.tar.gz
