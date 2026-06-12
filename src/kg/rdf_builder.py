from __future__ import annotations

from decimal import Decimal
from typing import Any

import pandas as pd
from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef, XSD

from src.canonical.schema import CANONICAL_DIR, make_hash, normalize_text, sanitize_code
from src.kg.rdf_store import INSTANCE_TTL, ensure_kg_dir


BI = Namespace("https://biz-insight.local/ontology#")


METRIC_CLASS = {
    "financial": BI.FinancialMetric,
    "credit": BI.CreditMetric,
    "investment": BI.InvestmentMetric,
    "employee": BI.EmployeeMetric,
    "stock": BI.StockMetric,
    "macro": BI.MacroMetric,
}


def _read(name: str) -> pd.DataFrame:
    path = CANONICAL_DIR / name
    return pd.read_csv(path, dtype=str) if path.exists() else pd.DataFrame()


def _resource(prefix: str, value: Any) -> URIRef:
    return BI[f"{prefix}_{sanitize_code(value)}"]


def _literal(value: Any, datatype: URIRef | None = None) -> Literal | None:
    text = normalize_text(value)
    if text is None:
        return None
    if datatype == XSD.decimal:
        try:
            Decimal(text)
        except Exception:
            return None
    return Literal(text, datatype=datatype)


def _add_literal(graph: Graph, subject: URIRef, predicate: URIRef, value: Any, datatype: URIRef | None = None) -> None:
    lit = _literal(value, datatype)
    if lit is not None:
        graph.add((subject, predicate, lit))


def build_graph(observation_limit: int | None = None) -> Graph:
    graph = Graph()
    graph.bind("bi", BI)
    graph.bind("xsd", XSD)
    graph.bind("rdfs", RDFS)

    companies = _read("companies.csv")
    aliases = _read("company_aliases.csv")
    sectors = _read("sectors.csv")
    metrics = _read("metrics.csv")
    observations = _read("observations.csv")
    ratings = _read("credit_ratings.csv")
    reviews = _read("employee_review_summaries.csv")
    criteria = _read("metric_criteria.csv")
    risk_signals = _read("risk_signals.csv")

    for row in sectors.to_dict("records"):
        uri = _resource("sector", row["sector_code"])
        graph.add((uri, RDF.type, BI.Sector))
        _add_literal(graph, uri, BI.sectorName, row.get("sector_name"))
        _add_literal(graph, uri, RDFS.label, row.get("sector_name"))

    for row in companies.to_dict("records"):
        uri = _resource("company", row["stock_code"])
        graph.add((uri, RDF.type, BI.Company))
        _add_literal(graph, uri, BI.stockCode, row.get("stock_code"))
        _add_literal(graph, uri, BI.preferredName, row.get("preferred_name"))
        _add_literal(graph, uri, RDFS.label, row.get("preferred_name"))
        if normalize_text(row.get("sector_code")):
            graph.add((uri, BI.belongsToSector, _resource("sector", row["sector_code"])))

    for row in aliases.to_dict("records"):
        alias_uri = BI[f"alias_{row['stock_code']}_{make_hash(row.get('alias_name'), length=10)}"]
        company_uri = _resource("company", row["stock_code"])
        graph.add((alias_uri, RDF.type, BI.CompanyAlias))
        _add_literal(graph, alias_uri, BI.aliasName, row.get("alias_name"))
        _add_literal(graph, alias_uri, RDFS.label, row.get("alias_name"))
        graph.add((company_uri, BI.hasAlias, alias_uri))

    for row in metrics.to_dict("records"):
        uri = _resource("metric", row["metric_code"])
        graph.add((uri, RDF.type, METRIC_CLASS.get(row.get("metric_category"), BI.Metric)))
        _add_literal(graph, uri, BI.metricCode, row.get("metric_code"))
        _add_literal(graph, uri, BI.metricNameKo, row.get("metric_name_ko"))
        _add_literal(graph, uri, BI.metricNameEn, row.get("metric_name_en"))
        _add_literal(graph, uri, BI.metricCategory, row.get("metric_category"))
        _add_literal(graph, uri, BI.unit, row.get("unit"))
        _add_literal(graph, uri, RDFS.label, row.get("metric_name_ko") or row.get("metric_name_en") or row.get("metric_code"))

    for row in criteria.to_dict("records"):
        metric_uri = _resource("metric", row["metric_code"])
        criterion_uri = _resource("criterion", row["criterion_code"])
        graph.add((metric_uri, BI.supportsCriterion, criterion_uri))
        _add_literal(graph, metric_uri, BI.evidenceDirection, row.get("evidence_direction"))

    for row in risk_signals.to_dict("records"):
        uri = _resource("risk_signal", row["risk_signal_code"])
        graph.add((uri, RDF.type, BI.RiskSignal))
        graph.add((uri, BI.relatedToCriterion, _resource("criterion", row["criterion_code"])))
        _add_literal(graph, uri, BI.riskSignalText, row.get("description"))
        _add_literal(graph, uri, RDFS.label, row.get("risk_signal_code"))

    if observation_limit is not None:
        observations = observations.head(observation_limit)
    for row in observations.to_dict("records"):
        uri = _resource("obs", row["observation_id"])
        subject_type = row.get("subject_type")
        if subject_type == "sector":
            obs_class = BI.SectorObservation
            subject_uri = _resource("sector", row["subject_id"])
            link = BI.hasSectorObservation
        elif subject_type == "macro":
            obs_class = BI.MacroObservation
            subject_uri = BI.macro_KR
            link = BI.hasMacroObservation
        else:
            obs_class = BI.CompanyObservation
            subject_uri = _resource("company", row["subject_id"])
            link = BI.hasObservation
        period_type = row.get("period_type")
        period_value = row.get("period_value")
        period_uri = _resource("fiscal_year" if period_type == "fiscal_year" else period_type, period_value)
        graph.add((uri, RDF.type, obs_class))
        graph.add((uri, BI.observedMetric, _resource("metric", row["metric_code"])))
        graph.add((uri, BI.observedPeriod, period_uri))
        graph.add((subject_uri, link, uri))
        graph.add((period_uri, RDF.type, BI.FiscalYear if period_type == "fiscal_year" else BI.TimePeriod))
        _add_literal(graph, period_uri, BI.periodValue, period_value)
        _add_literal(graph, uri, BI.numericValue, row.get("numeric_value"), XSD.decimal)
        _add_literal(graph, uri, BI.textValue, row.get("text_value"))
        _add_literal(graph, uri, BI.unit, row.get("unit"))

    for row in ratings.to_dict("records"):
        uri = _resource("rating", row["rating_id"])
        company_uri = _resource("company", row["stock_code"])
        period_uri = _resource("fiscal_year", row["year"])
        agency_uri = BI[sanitize_code(row["agency"]) or "Model"]
        graph.add((uri, RDF.type, BI.CreditRating))
        graph.add((company_uri, BI.hasCreditRating, uri))
        graph.add((uri, BI.ratingPeriod, period_uri))
        graph.add((uri, BI.ratedBy, agency_uri))
        _add_literal(graph, uri, BI.ratingValue, row.get("rating_value"))
        _add_literal(graph, uri, BI.predictedRatingValue, row.get("predicted_rating_value"))
        _add_literal(graph, uri, BI.bondType, row.get("bond_type"))

    for row in reviews.to_dict("records"):
        uri = _resource("review_summary", row["summary_id"])
        company_uri = _resource("company", row["stock_code"])
        period_uri = _resource(row["period_type"], row["period_value"])
        graph.add((uri, RDF.type, BI.EmployeeReview))
        graph.add((company_uri, BI.hasEmployeeReview, uri))
        graph.add((uri, BI.reviewPeriod, period_uri))
        _add_literal(graph, uri, BI.reviewTextPositive, row.get("positive_summary"))
        _add_literal(graph, uri, BI.reviewTextNegative, row.get("negative_summary"))
        _add_literal(graph, uri, BI.numericValue, row.get("rating"), XSD.decimal)

    return graph


def export_graph(observation_limit: int | None = None) -> str:
    ensure_kg_dir()
    graph = build_graph(observation_limit=observation_limit)
    graph.serialize(destination=str(INSTANCE_TTL), format="turtle")
    print(f"exported: {INSTANCE_TTL}")
    print(f"triples: {len(graph)}")
    return str(INSTANCE_TTL)

