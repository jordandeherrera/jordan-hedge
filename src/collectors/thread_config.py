"""
Thread-to-email mapping configuration.
Defines which Gmail queries and sender/subject patterns map to which threads and signals.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ThreadEmailConfig:
    """Gmail signal configuration for a thread."""
    thread_name: str
    domain: str

    # Gmail search queries that indicate activity on this thread
    search_queries: list[str] = field(default_factory=list)

    # Sender domains/addresses that are relevant
    relevant_senders: list[str] = field(default_factory=list)

    # Subject keywords that indicate relevance
    subject_keywords: list[str] = field(default_factory=list)

    # Senders that are adversarial/counterparty (triggers counterparty_active)
    adversarial_senders: list[str] = field(default_factory=list)

    # If outbound email to these addresses goes unanswered > N days
    expected_reply_from: list[str] = field(default_factory=list)
    reply_expected_within_days: int = 7


THREAD_CONFIGS: list[ThreadEmailConfig] = [

    ThreadEmailConfig(
        thread_name="MetLife Policy Investigation",
        domain="estate",
        search_queries=[
            "from:metlife.com OR from:metlife.org",
            "subject:(metlife OR \"policy 922300140\" OR \"estate of melvin\" OR \"DeHerrera estate\")",
        ],
        relevant_senders=["metlife.com", "metlife.org"],
        subject_keywords=["metlife", "policy 922300140", "estate of melvin", "deherrera estate",
                          "life insurance", "death benefit", "beneficiary"],
        expected_reply_from=["metlife.com", "metlife.org"],
        reply_expected_within_days=30,
    ),

    ThreadEmailConfig(
        thread_name="Estate Inventory Deadline",
        domain="estate",
        search_queries=[
            "from:(troy OR anderson OR attorney) subject:(estate OR inventory OR appraisal OR pueblo)",
            "subject:(24PR316 OR \"estate of melvin\" OR \"capistrano\" OR \"mahren\" OR appraisal)",
        ],
        relevant_senders=["troyandersonlaw.com", "pueblocounty.us", "courthouse"],
        subject_keywords=["inventory", "appraisal", "24pr316", "estate of melvin",
                          "capistrano", "mahren", "pueblo", "probate", "letters testamentary"],
        expected_reply_from=["troyandersonlaw.com"],
        reply_expected_within_days=7,
    ),

    ThreadEmailConfig(
        thread_name="Estate Fraud Investigation",
        domain="estate",
        search_queries=[
            "subject:(doreen OR sarah OR fraud OR \"death certificate\" OR \"hospital document\")",
            "from:(doreen OR sarah) subject:(estate OR insurance OR certificate)",
        ],
        relevant_senders=[],
        subject_keywords=["doreen", "sarah", "death certificate", "hospital document",
                          "patient advocacy", "fraud", "unauthorized"],
        adversarial_senders=["doreen", "sarah"],
    ),

    ThreadEmailConfig(
        thread_name="Medical Malpractice — Renown",
        domain="malpractice",
        search_queries=[
            "from:(mowbray OR nomura OR renown OR renownhealth)",
            "subject:(malpractice OR settlement OR renown OR \"ischial\" OR palacio OR \"personal injury\")",
            "from:mychart@renown.org",
        ],
        relevant_senders=["mowbraylaw.com", "nomuralegal.com", "renown.org", "renownhealth.org"],
        subject_keywords=["malpractice", "settlement", "renown", "ischial", "palacio",
                          "personal injury", "medical records", "mychart"],
        adversarial_senders=["renown.org", "renownhealth.org"],
        expected_reply_from=["mowbraylaw.com", "nomuralegal.com"],
        reply_expected_within_days=7,
    ),

    ThreadEmailConfig(
        thread_name="MEL Cloud Deployment",
        domain="MEL",
        search_queries=[
            "from:github.com subject:(patient-talk OR mel-backend OR deploy)",
            "subject:(aws OR ecs OR deployment OR \"patient talk\" OR mel)",
        ],
        relevant_senders=["github.com", "amazonaws.com", "aws.amazon.com"],
        subject_keywords=["patient-talk-summary", "mel-backend", "ecs", "deployment",
                          "aws", "github actions", "build failed", "build passed"],
    ),

    ThreadEmailConfig(
        thread_name="Meniscus Tear Recovery",
        domain="health",
        search_queries=[
            "from:(swift OR sportsdome OR ryanmaves OR bosque OR ways2well)",
            "subject:(meniscus OR physical therapy OR knee OR peptide OR bpc OR \"tb-500\")",
            "from:mychart",
        ],
        relevant_senders=["swiftpt.com", "bosqueclinic.com", "ways2well.com", "mychart"],
        subject_keywords=["meniscus", "physical therapy", "knee", "peptide", "bpc-157",
                          "tb-500", "appointment", "ryan maves", "orthopedic"],
        expected_reply_from=["bosqueclinic.com", "ways2well.com"],
        reply_expected_within_days=5,
    ),

    ThreadEmailConfig(
        thread_name="UnitedHealthcare Benefits Optimization",
        domain="finance",
        search_queries=[
            "from:(uhc OR unitedhealthcare OR optumrx OR rally)",
            "subject:(claim OR EOB OR \"explanation of benefits\" OR UHC OR FSA OR HSA)",
        ],
        relevant_senders=["uhc.com", "unitedhealthcare.com", "optumrx.com", "rally.com"],
        subject_keywords=["claim denied", "prior authorization", "eob", "explanation of benefits",
                          "fsa", "hsa", "optumrx", "rally rewards"],
    ),

    ThreadEmailConfig(
        thread_name="Ridgeline L6 Promotion",
        domain="ridgeline",
        search_queries=[
            "from:ridgelineapps.com subject:(review OR promotion OR performance OR L6 OR principal)",
        ],
        relevant_senders=["ridgelineapps.com", "latticehq.com"],
        subject_keywords=["performance review", "promotion", "principal", "l6",
                          "lattice", "360", "feedback"],
    ),
]

# Map thread name → config for fast lookup
THREAD_CONFIG_MAP: dict[str, ThreadEmailConfig] = {
    c.thread_name: c for c in THREAD_CONFIGS
}
