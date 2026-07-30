import pytest

from mlem_method.rsa import RSA


def test_rsa_job_array_canonicalizes_model_layer_pairs() -> None:
    """Job-array clones share a UID while layers remain attached to their models."""
    rsa = RSA()
    pairs = (
        (("model-b", 3), ("model-a", 7)),
        (("model-a", 7), ("model-b", 3)),
    )

    # Enter a real job-array context, but stop after task creation to avoid running RSA.
    with pytest.raises(RuntimeError, match="stop"), rsa.layers_infra.job_array() as jobs:
        for (left, left_layer), (right, right_layer) in pairs:
            jobs.append(
                rsa.layers_infra.clone_obj(
                    representations_1={
                        "level": "sentence",
                        "model_name": left,
                        "layer": left_layer,
                    },
                    representations_2={
                        "level": "sentence",
                        "model_name": right,
                        "layer": right_layer,
                    },
                )
            )

        assert [
            (
                job.representations_1.model_dump()["model_name"],
                job.representations_1.model_dump()["layer"],
                job.representations_2.model_dump()["model_name"],
                job.representations_2.model_dump()["layer"],
            )
            for job in jobs
        ] == [("model-a", 7, "model-b", 3)] * 2
        assert jobs[0].infra.uid() == jobs[1].infra.uid()
        raise RuntimeError("stop")  # avoid submitting the test jobs
