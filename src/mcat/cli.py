"""``mcat``: run pyAFQ (for its derivatives) and MCAT on one participant."""
import argparse
import json
import logging


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dwi"), p.add_argument("bval"), p.add_argument("bvec")
    p.add_argument("t1", help="T1w image, registered to the DWI (pyAFQ's t1_file)")
    p.add_argument("output_dir")
    p.add_argument("--n-proc", type=int, required=True, help="threads")
    p.add_argument("--max-mem", required=True, help="memory budget, e.g. 16GB")
    p.add_argument("--moments", action="store_true", help="spatial first moments (narrower bundles)")
    p.add_argument("--tol", type=float, default=1e-10, help="remaining walker mass at which solves stop")
    p.add_argument("--angle", type=float, default=35.0, help="maximum turn per one-voxel hop (deg)")
    p.add_argument("--single-odf", action="store_true", help="first-order hops (pyAFQ-like), not doubleODF")
    p.add_argument("--bundles", nargs="+", help="pyAFQ bundle names (default: all)")
    p.add_argument("--afq-kwargs", default="{}", help="JSON of extra ParticipantAFQ arguments")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    from AFQ.api.participant import ParticipantAFQ

    from .afq import run

    kwargs = json.loads(a.afq_kwargs)
    if "tracking_params" in kwargs:
        if "odf_model" not in kwargs["tracking_params"]:
            kwargs["tracking_params"]["odf_model"] = "CSD"
    else:
        kwargs["tracking_params"] = {"odf_model": "CSD"}

    afq = ParticipantAFQ(a.dwi, a.bval, a.bvec, a.t1, a.output_dir, **kwargs)
    run(afq, a.n_proc, a.max_mem, a.bundles, a.moments, a.tol, a.angle, not a.single_odf)


if __name__ == "__main__":
    main()
