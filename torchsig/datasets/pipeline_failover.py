# ruff: noqa: PERF203
import logging
import sys
import traceback
from typing import Literal

import numpy as np

log = logging.getLogger(__name__)

handler = logging.StreamHandler(sys.stdout)
log.addHandler(handler)
log.setLevel(logging.WARNING)


class PipelineFailOverEnabled:
    """Simple mix-in that gives the dataset a fallback behaviour.
    For *any* exception that occurs during the transform pipeline the
    dataset can:

    * return the original data unchanged (`fallback="original"`).
    * return a zero-array of identical shape (`fallback="zero"`).
    * repeat the whole step up to `max_retries` times and, if all
      attempts fail, fall back to the *original* data (`fallback="retry"`).

    The behaviour is controlled by two attributes:

    * `pipeline_fallback` : {"original", "zero", "retry"}
    * `pipeline_max_retries`: int   (must be >0 for “retry”)
    """

    pipeline_fallback: Literal["original", "zero", "retry"] = "original"
    pipeline_max_retries: int = 1

    def _run_with_fallback(self, func, *args, fallback_raw_signal=None, **kwargs):
        """Execute a pipeline stage with retry and fallback handling.

        Calls ``func`` with the provided arguments and applies the configured
        pipeline recovery policy if an exception occurs.

        When ``pipeline_fallback == "retry"``, the function is retried up to
        ``pipeline_max_retries`` times. All failed attempts are logged along
        with the exception type and traceback.

        If all retry attempts are exhausted and ``fallback_raw_signal`` is
        provided, the configured fallback action is applied to that signal via
        :meth:`_fallback_action`. If no fallback signal is available, the final
        exception is re-raised.

        Args:
            func: Callable representing a pipeline stage.
            *args: Positional arguments passed to ``func``.
            fallback_raw_signal: Raw signal to use for fallback recovery if the
                stage ultimately fails. If ``None``, no fallback is available and
                the exception is propagated.
            **kwargs: Keyword arguments passed to ``func``.

        Returns:
            The result of ``func`` if execution succeeds, or the result of the
            configured fallback action if recovery is triggered.

        Raises:
            Exception: Re-raises the final exception when retries are exhausted
                and no fallback signal is available.
        """
        retries = self.pipeline_max_retries if self.pipeline_fallback == "retry" else 1

        for attempt in range(retries):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                log.warning(f"Pipeline retry {attempt + 1}/{retries} failed: {type(exc).__name__}: {exc}\nTraceback:\n{traceback.format_exc()}")

                if attempt == retries - 1:
                    log.warning("Retries exhausted.")

                    if fallback_raw_signal is not None:
                        return self._fallback_action(fallback_raw_signal)

                    raise

    def _fallback_action(self, raw_signal):
        """Return a safe tensor that does *not* depend on the broken transform.

        The returned array matches the shape of ``raw_signal.data``.
        """
        if self.pipeline_fallback == "original":
            # No transforms applied -- just return the raw data.
            return raw_signal.data

        if self.pipeline_fallback == "zero":
            return np.zeros_like(raw_signal.data, dtype=raw_signal.data.dtype)

        if self.pipeline_fallback == "retry":
            # After exhausting retries we treat the behaviour as “original”.
            return raw_signal.data

        # Anything else is a programming error.
        raise RuntimeError(f"Unknown fallback option: {self.pipeline_fallback!r}")
