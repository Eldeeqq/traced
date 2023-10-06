import json
from abc import abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from matplotlib.axes import Axes
from tqdm.auto import tqdm
from traitlets import Callable

from traced_v2.models.base_model import BaseModel, Visual
from traced_v2.models.bernoulli import BernoulliModelOutput
from traced_v2.models.poisson import PoissonModel
from traced_v2.trace_analyzer import (MultiTraceAnalyzer, TraceAnalyzer,
                                      TraceAnalyzerOutput)


class SiteAnalyzer(BaseModel, Visual):
    def __init__(self, src: str, dest: str, *args, **kwargs):
        super().__init__(src, dest, *args, **kwargs)
        self.site_to_site: dict[str, dict[str, MultiTraceAnalyzer]] = {}
        self.n_anomalies: PoissonModel = PoissonModel(src, dest, parent=self)

    def log(self, data) -> tuple[None | TraceAnalyzerOutput, BernoulliModelOutput]:
        ts = data["timestamp"]
        super().log_timestamp(ts)

        src_site = data["src_site"]
        dest_site = data["dest_site"]

        model: MultiTraceAnalyzer | None = None
        if src_site not in self.site_to_site:
            model = MultiTraceAnalyzer(src_site, dest_site)
            self.site_to_site[src_site] = {dest_site: model}
        else:
            if dest_site not in self.site_to_site[src_site]:
                model = MultiTraceAnalyzer(src_site, dest_site)
                self.site_to_site[src_site][dest_site] = model
            else:
                model = self.site_to_site[src_site][dest_site]

        return model.log(data)  # type: ignore

    def to_dict(self):
        return self.n_anomalies.to_dict()

    def plot(self, ax: Axes | None = None, **kwargs) -> None:
        self.n_anomalies.plot(ax, **kwargs)

    def process_stream(self, stream):
        """Placeholder for processing a stream of data."""
        raise NotImplementedError("TODO")

    def process_files(
        self,
        files: list[str | Path],
        show_progress: bool = False,
        filter: None | Callable = None,
        **kwargs,
    ):  # type: ignore
        """Process a list of files."""
        paths = [Path(x) if isinstance(x, str) else x for x in files]

        if show_progress:
            paths = tqdm(
                paths,
                desc="Processing files",
                unit="files",
                total=len(files),
                leave=True,
            )

        for path in paths:
            with path.open("r") as f:
                data = json.load(f)
            if filter is not None:
                if not filter(data):  # type: ignore
                    continue
            out = self.log(data)
            # TODO: add result processing here

    def process_folder(
        self,
        directory: Path | str,
        show_progress: bool = False,
        filter: None | Callable = None,
        **kwargs,
    ) -> None:
        directory = directory if isinstance(directory, Path) else Path(directory)
        assert directory.is_dir()

        subdirs = [x for x in directory.glob("*") if x.is_dir()]
        with ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="SiteAnalyzer"
        ) as executor:
            result = list(
                tqdm(
                    executor.map(
                        self.process_site,
                        subdirs,
                        [show_progress] * len(subdirs),
                        [filter] * len(subdirs),
                    ),
                    desc=f"Datasets processed",
                    unit="files",
                    total=len(subdirs),
                    position=0,
                    leave=True,
                )
            )
        for r in result:
            for src, data in r.items():
                if src not in self.site_to_site:
                    self.site_to_site[src] = {}
                for dest, model in data.items():
                    self.site_to_site[src][dest] = model  # type: ignore

    @classmethod
    def process_site(
        cls, folder: Path, show_progress: bool = False, filter: None | Callable = None
    ) -> dict[str, dict[str, None | MultiTraceAnalyzer]]:
        files = sorted(
            list(folder.rglob("*.json"))
        )  # assuming there is timestamp in the filename
        analyzers: dict[str, dict[str, None | MultiTraceAnalyzer]] = defaultdict(
            lambda: defaultdict(lambda: None)
        )

        checked = False
        if folder.name.split("-") == 2:
            if not filter(*folder.name.split("_")):  # type: ignore
                return analyzers
            checked = True

        if show_progress:
            files = tqdm(
                files, desc=f"folder {str(folder)}", unit="files", total=len(files)
            )

        for file in files:
            with file.open("r") as f:
                data = json.load(f)

            src_site = data.get("src_site", "")
            dest_site = data.get("dest_site", "")

            if filter is not None and not checked:
                if not filter(src_site, dest_site):  # type: ignore
                    continue

            if not analyzers[src_site][dest_site]:
                analyzers[src_site][dest_site] = MultiTraceAnalyzer(src_site, dest_site)

            analyzers[src_site][dest_site].log(data)  # type: ignore

        return dict(analyzers)
