"""Filesystem workspace for paper manifests, recipes, implementations, and tests."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

from core.config import deep_merge, load_config, save_config
from core.extensions import extension_paths_from_config
from experiments import normalize_experiment_config
from reproduction.specs import PaperConfigError, PaperSpec, RecipeSpec


PAPER_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class PaperWorkspace:
    def __init__(
        self,
        root: str | Path = "paper_reproductions",
        project_root: str | Path | None = None,
    ):
        self.project_root = Path(project_root or Path.cwd()).expanduser().resolve()
        root_path = Path(root).expanduser()
        if not root_path.is_absolute():
            root_path = self.project_root / root_path
        self.root = root_path.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def validate_paper_id(paper_id: str) -> str:
        normalized = paper_id.strip().lower()
        if not PAPER_ID_PATTERN.fullmatch(normalized):
            raise PaperConfigError(
                "paper id must start with a lowercase letter or digit and contain "
                "only lowercase letters, digits, '.', '_' or '-'"
            )
        return normalized

    def paper_directory(self, paper_id: str) -> Path:
        return self.root / self.validate_paper_id(paper_id)

    @staticmethod
    def validate_recipe_name(recipe_name: str) -> str:
        if not PAPER_ID_PATTERN.fullmatch(recipe_name):
            raise PaperConfigError(
                "recipe name must start with a lowercase letter or digit and contain "
                "only lowercase letters, digits, '.', '_' or '-'"
            )
        return recipe_name

    def scaffold(
        self,
        paper_id: str,
        title: str,
        *,
        url: str | None = None,
        authors: list[str] | None = None,
        year: int | None = None,
    ) -> Path:
        paper_id = self.validate_paper_id(paper_id)
        if not title.strip():
            raise PaperConfigError("paper title must be non-empty")
        directory = self.paper_directory(paper_id)
        if directory.exists():
            raise FileExistsError(f"Paper workspace already exists: {directory}")
        recipes = directory / "recipes"
        implementation = directory / "implementation"
        tests = directory / "tests"
        notes = directory / "notes"
        for path in (recipes, implementation, tests, notes):
            path.mkdir(parents=True, exist_ok=False)

        manifest = {
            "schema_version": 1,
            "paper": {
                "id": paper_id,
                "title": title.strip(),
                "authors": list(authors or []),
                "year": year,
                "url": url,
                "venue": None,
                "tags": [],
            },
            "implementation": {
                "status": "scaffold",
                "upstream_repository": None,
                "upstream_revision": None,
            },
            "claims": [
                {
                    "id": "harness_smoke",
                    "statement": "Both scaffold recipes complete successfully.",
                    "source": "Local harness check; replace with a paper claim.",
                    "expectations": [
                        {
                            "type": "absolute",
                            "recipe": "baseline_smoke",
                            "metric": "status",
                            "operator": "==",
                            "value": "completed",
                        },
                        {
                            "type": "absolute",
                            "recipe": "proposed_smoke",
                            "metric": "status",
                            "operator": "==",
                            "value": "completed",
                        },
                    ],
                }
            ],
            "suites": {
                "smoke": {
                    "recipes": ["baseline_smoke", "proposed_smoke"],
                    "claims": ["harness_smoke"],
                    "strategy": "sequential",
                }
            },
        }
        save_config(manifest, directory / "paper.yaml")
        for recipe_name in ("baseline_smoke", "proposed_smoke"):
            save_config(
                {
                    "schema_version": 1,
                    "name": recipe_name,
                    "description": (
                        f"Scaffold {recipe_name} recipe; replace its overrides."
                    ),
                    "base_config": "configs/smoke.yaml",
                    "overrides": {
                        "experiment": {
                            "name": f"{paper_id}-{recipe_name}",
                        }
                    },
                },
                recipes / f"{recipe_name}.yaml",
            )
        (implementation / "__init__.py").write_text("", encoding="utf-8")
        (implementation / "components.py").write_text(
            '"""Paper-specific registered components and experiment stages."""\n\n'
            "# Register only the technique introduced by the paper. Reuse the\n"
            "# framework for models, training, evaluation, and execution.\n",
            encoding="utf-8",
        )
        (tests / "__init__.py").write_text("", encoding="utf-8")
        (tests / "README.md").write_text(
            "# Paper tests\n\n"
            "Add correctness tests for the paper-specific implementation before "
            "running full-scale recipes.\n",
            encoding="utf-8",
        )
        (notes / "reading.md").write_text(
            f"# {title.strip()}\n\n"
            "## Problem\n\nTODO\n\n"
            "## Method\n\nTODO\n\n"
            "## Claims to verify\n\nTODO\n\n"
            "## Reproduction differences\n\nTODO\n",
            encoding="utf-8",
        )
        (directory / "README.md").write_text(
            self._paper_readme(paper_id, title.strip()), encoding="utf-8"
        )
        return directory

    @staticmethod
    def _paper_readme(paper_id: str, title: str) -> str:
        return (
            f"# {title}\n\n"
            "This directory contains only the local reproduction contract, "
            "paper-specific code, tests, and notes. Upstream source belongs in "
            "`external_projects/`.\n\n"
            "```bash\n"
            f"python3 -m cli paper validate {paper_id}\n"
            f"python3 -m cli paper study {paper_id} --suite smoke\n"
            "```\n\n"
            "Add an implementation file to a recipe with:\n\n"
            "```yaml\n"
            "overrides:\n"
            "  extensions:\n"
            "    paths:\n"
            f"      - paper_reproductions/{paper_id}/implementation/components.py\n"
            "```\n"
        )

    def list(self) -> list[PaperSpec]:
        papers = []
        for manifest_path in sorted(self.root.glob("*/paper.yaml")):
            papers.append(PaperSpec.from_dict(load_config(manifest_path)))
        return papers

    def load(self, paper_id: str) -> PaperSpec:
        path = self.paper_directory(paper_id) / "paper.yaml"
        if not path.exists():
            raise KeyError(f"Unknown paper id: {paper_id}")
        spec = PaperSpec.from_dict(load_config(path))
        if spec.id != self.validate_paper_id(paper_id):
            raise PaperConfigError(
                f"paper.id '{spec.id}' does not match directory '{paper_id}'"
            )
        return spec

    def load_recipe(self, paper_id: str, recipe_name: str) -> RecipeSpec:
        recipe_name = self.validate_recipe_name(recipe_name)
        path = self.paper_directory(paper_id) / "recipes" / f"{recipe_name}.yaml"
        if not path.exists():
            raise KeyError(f"Unknown recipe '{recipe_name}' for paper '{paper_id}'")
        recipe = RecipeSpec.from_dict(load_config(path), str(path))
        if recipe.name != recipe_name:
            raise PaperConfigError(
                f"Recipe name '{recipe.name}' does not match filename '{recipe_name}'"
            )
        return recipe

    def _config_path(self, paper_id: str, value: str) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        project_candidate = self.project_root / path
        if project_candidate.exists():
            return project_candidate
        return self.paper_directory(paper_id) / path

    def resolve_recipe(self, paper_id: str, recipe_name: str) -> dict[str, Any]:
        paper = self.load(paper_id)
        recipe = self.load_recipe(paper_id, recipe_name)
        config: Mapping[str, Any] = {}
        if recipe.base_config:
            config_path = self._config_path(paper_id, recipe.base_config)
            if not config_path.exists():
                raise FileNotFoundError(
                    f"Base config for recipe '{recipe_name}' does not exist: "
                    f"{config_path}"
                )
            config = load_config(config_path)
        resolved = deep_merge(config, recipe.config)
        resolved = deep_merge(resolved, recipe.overrides)
        if recipe.execution:
            resolved = deep_merge(resolved, {"execution": recipe.execution})
        manifest_path = self.paper_directory(paper_id) / "paper.yaml"
        try:
            manifest_reference = str(manifest_path.relative_to(self.project_root))
        except ValueError:
            manifest_reference = str(manifest_path)
        resolved["paper_reproduction"] = {
            "paper_id": paper.id,
            "title": paper.title,
            "recipe": recipe.name,
            "manifest": manifest_reference,
            "url": paper.url,
        }
        return normalize_experiment_config(resolved)

    def validate(self, paper_id: str) -> dict[str, Any]:
        paper = self.load(paper_id)
        recipe_names = {
            path.stem
            for path in (self.paper_directory(paper_id) / "recipes").glob("*.yaml")
        }
        referenced = {name for suite in paper.suites.values() for name in suite.recipes}
        missing = referenced - recipe_names
        if missing:
            raise PaperConfigError(
                f"Suites reference missing recipes: {', '.join(sorted(missing))}"
            )
        for recipe_name in sorted(referenced):
            resolved = self.resolve_recipe(paper_id, recipe_name)
            for extension_path in extension_paths_from_config(resolved):
                path = Path(extension_path).expanduser()
                if not path.is_absolute():
                    path = self.project_root / path
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Recipe '{recipe_name}' extension does not exist: {path}"
                    )
        for claim in paper.claims:
            for expectation in claim.expectations:
                if expectation.recipe not in recipe_names:
                    raise PaperConfigError(
                        f"Claim '{claim.id}' references unknown recipe "
                        f"'{expectation.recipe}'"
                    )
                if expectation.baseline and expectation.baseline not in recipe_names:
                    raise PaperConfigError(
                        f"Claim '{claim.id}' references unknown baseline "
                        f"'{expectation.baseline}'"
                    )
        claims_by_id = {claim.id: claim for claim in paper.claims}
        for suite in paper.suites.values():
            selected_claims = (
                [claims_by_id[claim_id] for claim_id in suite.claims]
                if suite.claims
                else list(paper.claims)
            )
            required_recipes = {
                recipe
                for claim in selected_claims
                for expectation in claim.expectations
                for recipe in (expectation.recipe, expectation.baseline)
                if recipe
            }
            unavailable = required_recipes - set(suite.recipes)
            if unavailable:
                raise PaperConfigError(
                    f"Suite '{suite.name}' cannot assess its claims without recipes: "
                    f"{', '.join(sorted(unavailable))}"
                )
        return {
            "paper_id": paper.id,
            "title": paper.title,
            "recipes": sorted(recipe_names),
            "suites": sorted(paper.suites),
            "claims": [claim.id for claim in paper.claims],
        }
