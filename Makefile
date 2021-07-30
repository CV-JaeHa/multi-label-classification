
.PHONY: run
run: build
	docker compose up -d

.PHONY: build
build:
	docker build -t multi_label_classification .

.PHONY: stop
stop:
	docker compose down