PY=python


.PHONY: build train gen shell clean


build:
	docker compose build


train:
	docker compose run --rm trainer


gen:
	docker compose run --rm generator


shell:
	docker compose run --rm trainer bash


clean:
	rm -rf artifacts/models/* artifacts/samples/*