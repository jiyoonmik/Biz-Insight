from __future__ import annotations

import argparse

from rdflib import Graph

from src.kg.rdf_builder import BI, export_graph
from src.kg.rdf_store import INSTANCE_TTL, ONTOLOGY_TTL


def validate() -> dict[str, int]:
    ontology = Graph()
    ontology.parse(str(ONTOLOGY_TTL), format="turtle")
    instances = Graph()
    instances.parse(str(INSTANCE_TTL), format="turtle")
    counts = {
        "ontology_triples": len(ontology),
        "instance_triples": len(instances),
        "companies": len(list(instances.subjects(predicate=None, object=BI.Company))),
        "observations": len(list(instances.subjects(predicate=None, object=BI.CompanyObservation))),
        "credit_ratings": len(list(instances.subjects(predicate=None, object=BI.CreditRating))),
        "review_summaries": len(list(instances.subjects(predicate=None, object=BI.EmployeeReview))),
    }
    for key, value in counts.items():
        print(f"{key}: {value}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--observation-limit", type=int, default=None)
    args = parser.parse_args()
    export_graph(observation_limit=args.observation_limit)
    if args.validate:
        validate()


if __name__ == "__main__":
    main()

