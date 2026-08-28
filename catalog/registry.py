"""Curated registry of real data.gov.sg datasets, spanning diverse agencies.

Every STRUCTURED entry's dataset_id was verified live against the real
datastore_search API (HTTP 200, success: true, non-empty fields) while
building this registry -- see the manual verification log in the PR
description / commit history for the exact curl calls. DOCUMENT entries
have no live datastore resource; their text lives under data/ and is
ingested by the RAG fallback instead.
"""

from schemas import DatasetEntry, DatasetKind

REGISTRY: list[DatasetEntry] = [
    # --- Housing ---
    DatasetEntry(
        dataset_id="d_ebc5ab87086db484f88045b47411ebc5",
        title="HDB Resale Flat Prices",
        agency="Housing & Development Board",
        description=(
            "Resale transaction prices of HDB flats from 1990 onwards, including "
            "town, flat type, floor area, storey range, and lease details."
        ),
        tags=["housing", "hdb", "resale", "property"],
        kind=DatasetKind.STRUCTURED,
        fields=[
            "month",
            "town",
            "flat_type",
            "block",
            "street_name",
            "storey_range",
            "floor_area_sqm",
            "flat_model",
            "lease_commence_date",
            "resale_price",
        ],
        source_url="https://data.gov.sg/datasets/d_ebc5ab87086db484f88045b47411ebc5/view",
    ),
    DatasetEntry(
        dataset_id="d_c9f57187485a850908655db0e8cfe651",
        title="Renting Out of Flats from Jan 2021",
        agency="Housing & Development Board",
        description=(
            "Approved applications for subletting of HDB flats from January 2021 "
            "onwards, including town, block, street, flat type and monthly rent."
        ),
        tags=["housing", "hdb", "rental", "property"],
        kind=DatasetKind.STRUCTURED,
        fields=["rent_approval_date", "town", "block", "street_name", "flat_type", "monthly_rent"],
        source_url="https://data.gov.sg/datasets/d_c9f57187485a850908655db0e8cfe651/view",
    ),
    DatasetEntry(
        dataset_id="d_a9d1c55a241344c2a9645891d650a522",
        title="Resident Households by Income and Type of Dwelling (Census 2020)",
        agency="Singapore Department of Statistics",
        description=(
            "Census 2020 breakdown of resident households by monthly household "
            "income from work and by type of dwelling (HDB, condo, landed, other)."
        ),
        tags=["demographics", "census", "household income", "housing", "singstat"],
        kind=DatasetKind.STRUCTURED,
        fields=[
            "Number",
            "Total",
            "HDBDwellings_Total",
            "HDBDwellings_1_and2_RoomFlats2",
            "HDBDwellings_3_RoomFlats",
            "HDBDwellings_4_RoomFlats",
            "HDBDwellings_5_RoomandExecutiveFlats",
            "CondominiumsandOtherApartments",
            "LandedProperties",
            "Others",
        ],
        source_url="https://data.gov.sg/datasets/d_a9d1c55a241344c2a9645891d650a522/view",
    ),
    DatasetEntry(
        dataset_id="d_8e4c50283fb7052a391dfb746a05c853",
        title="Private Residential Property Rental Index, Quarterly",
        agency="Urban Redevelopment Authority",
        description=(
            "Quarterly rental index for private residential properties by property "
            "type and locality, base quarter 2009-Q1 = 100."
        ),
        tags=["housing", "property", "rental", "real estate", "ura"],
        kind=DatasetKind.STRUCTURED,
        fields=["quarter", "property_type", "locality", "index"],
        source_url="https://data.gov.sg/datasets/d_8e4c50283fb7052a391dfb746a05c853/view",
    ),
    DatasetEntry(
        dataset_id="d_97f8a2e995022d311c6c68cfda6d034c",
        title="Private Residential Property Price Index, Quarterly",
        agency="Urban Redevelopment Authority",
        description=(
            "Quarterly price index for private residential properties by property "
            "type, base quarter 2009-Q1 = 100."
        ),
        tags=["housing", "property", "price index", "real estate", "ura"],
        kind=DatasetKind.STRUCTURED,
        fields=["quarter", "property_type", "index"],
        source_url="https://data.gov.sg/datasets/d_97f8a2e995022d311c6c68cfda6d034c/view",
    ),
    # --- Transport ---
    DatasetEntry(
        dataset_id="d_778e6d2eaf4a3812aab0d1a1bdf7fd38",
        title="Commuter Facilities",
        agency="Land Transport Authority",
        description=(
            "Annual counts of bus stops, bus interchanges, bus terminals, taxi "
            "stops and taxi stands maintained by LTA."
        ),
        tags=["transport", "bus", "taxi", "infrastructure"],
        kind=DatasetKind.STRUCTURED,
        fields=["year", "facility", "number"],
        source_url="https://data.gov.sg/datasets/d_778e6d2eaf4a3812aab0d1a1bdf7fd38/view",
    ),
    DatasetEntry(
        dataset_id="d_be2accb464cc5600de937eb9000a0255",
        title="Premium Bus Services",
        agency="Land Transport Authority",
        description=(
            "Route-level information for Premium Bus Service stops, including "
            "operator, direction, first/last bus times, origin-destination, and fares."
        ),
        tags=["transport", "bus", "premium bus", "routes"],
        kind=DatasetKind.STRUCTURED,
        fields=[
            "opr_desc_txt",
            "bus_svc_no_txt",
            "bus_dirctn_txt",
            "bus_route_seq_num",
            "rd_nam",
            "bus_stop_desc_txt",
            "op_hr_1_txt",
            "op_hr_2_txt",
            "orig_dest_txt",
            "fare_txt",
            "bus_stop_cd",
        ],
        source_url="https://data.gov.sg/datasets/d_be2accb464cc5600de937eb9000a0255/view",
    ),
    DatasetEntry(
        dataset_id="d_d6921f812624c2b1bb1d68269354dc71",
        title="Public Transport Journeys",
        agency="Land Transport Authority",
        description=(
            "Annual average daily passenger journeys and average journey distances "
            "across Singapore's public transport network."
        ),
        tags=["transport", "public transport", "ridership"],
        kind=DatasetKind.STRUCTURED,
        fields=["year", "average_daily_passenger_journeys", "average_journey_distances"],
        source_url="https://data.gov.sg/datasets/d_d6921f812624c2b1bb1d68269354dc71/view",
    ),
    # --- Environment ---
    DatasetEntry(
        dataset_id="d_9213cd2e4631f7148ab5932a10df9958",
        title="Historical Pollutant Standards Index (PSI) 2024",
        agency="National Environment Agency",
        description=(
            "Historical 2024 readings of PSI and component pollutant sub-indices "
            "(PM10, PM2.5, O3, CO, SO2, NO2) by region across Singapore."
        ),
        tags=["environment", "air quality", "psi", "pollution"],
        kind=DatasetKind.STRUCTURED,
        fields=[
            "date",
            "timestamp",
            "region",
            "pm10_twenty_four_hourly",
            "pm10_sub_index",
            "pm25_twenty_four_hourly",
            "pm25_sub_index",
            "o3_eight_hour_max",
            "psi_twenty_four_hourly",
        ],
        source_url="https://data.gov.sg/datasets/d_9213cd2e4631f7148ab5932a10df9958/view",
    ),
    DatasetEntry(
        dataset_id="d_b4cf557f8750260d229c49fd768e11ed",
        title="Historical 24-hr PSI",
        agency="National Environment Agency",
        description=(
            "Historical 24-hour Pollutant Standards Index readings broken down by "
            "the five regions of Singapore (North, South, East, West, Central)."
        ),
        tags=["environment", "air quality", "psi", "pollution"],
        kind=DatasetKind.STRUCTURED,
        fields=["24hr_psi", "north", "south", "east", "west", "central"],
        source_url="https://data.gov.sg/datasets/d_b4cf557f8750260d229c49fd768e11ed/view",
    ),
    DatasetEntry(
        dataset_id="d_50830e6a799ccfd00b1ed0a5294920d7",
        title="NEWater Tariff",
        agency="PUB, Singapore's National Water Agency",
        description=(
            "NEWater tariff rates by tariff category and consumption block, before "
            "and after GST, including the Waterborne Fee."
        ),
        tags=["environment", "water", "newater", "utilities", "tariff"],
        kind=DatasetKind.STRUCTURED,
        fields=[
            "tariff_category",
            "consumption_block",
            "tariff_before_gst",
            "tariff_after_gst",
            "wbf_before_gst",
            "wbf_after_gst",
        ],
        source_url="https://data.gov.sg/datasets/d_50830e6a799ccfd00b1ed0a5294920d7/view",
    ),
    DatasetEntry(
        dataset_id="d_2eceeb792a0fca1caa74304d47b46060",
        title="Volume of NEWater Sold, Annual",
        agency="PUB, Singapore's National Water Agency",
        description="Annual volume of NEWater sold in Singapore.",
        tags=["environment", "water", "newater", "utilities"],
        kind=DatasetKind.STRUCTURED,
        fields=["year", "sale_of_newater"],
        source_url="https://data.gov.sg/datasets/d_2eceeb792a0fca1caa74304d47b46060/view",
    ),
    # --- Health ---
    DatasetEntry(
        dataset_id="d_ac1eecf0886ff0bceefbc51556247015",
        title="Weekly Number of Dengue and Dengue Haemorrhagic Fever Cases",
        agency="Ministry of Health",
        description=(
            "Weekly count of laboratory-confirmed dengue and dengue haemorrhagic "
            "fever cases by epidemiological week and year."
        ),
        tags=["health", "dengue", "infectious disease", "public health"],
        kind=DatasetKind.STRUCTURED,
        fields=["year", "eweek", "type_dengue", "number"],
        source_url="https://data.gov.sg/datasets/d_ac1eecf0886ff0bceefbc51556247015/view",
    ),
    DatasetEntry(
        dataset_id="d_ca168b2cb763640d72c4600a68f9909e",
        title="Weekly Infectious Disease Bulletin Cases",
        agency="Ministry of Health",
        description=(
            "Weekly statistics on locally notifiable infectious diseases, "
            "published each epidemiological week (Sunday to Saturday)."
        ),
        tags=["health", "infectious disease", "epidemiology", "public health"],
        kind=DatasetKind.STRUCTURED,
        fields=["epi_week", "disease", "no._of_cases"],
        source_url="https://data.gov.sg/datasets/d_ca168b2cb763640d72c4600a68f9909e/view",
    ),
    DatasetEntry(
        dataset_id="d_e4663ad3f088a46dabd3972dc166402d",
        title="Health Facilities (Primary Care, Dental Clinics and Pharmacies)",
        agency="Ministry of Health",
        description=(
            "Annual counts of primary care institutions, dental clinics and "
            "pharmacies broken down by sector and facility type."
        ),
        tags=["health", "healthcare facilities", "primary care"],
        kind=DatasetKind.STRUCTURED,
        fields=["year", "institution_type", "sector", "facility_type_b", "no_of_facilities"],
        source_url="https://data.gov.sg/datasets/d_e4663ad3f088a46dabd3972dc166402d/view",
    ),
    # --- Economy / labour ---
    DatasetEntry(
        dataset_id="d_ce0511e418694cd265a3a6fc8c266782",
        title="Foreign Workforce Excluding FDW and Construction/Marine/Process WPH",
        agency="Ministry of Manpower",
        description=(
            "Monthly count of the foreign workforce in Singapore, excluding "
            "foreign domestic workers and Work Permit Holders in construction, "
            "marine, and process sectors."
        ),
        tags=["economy", "employment", "foreign workforce", "labour"],
        kind=DatasetKind.STRUCTURED,
        fields=["month", "count"],
        source_url="https://data.gov.sg/datasets/d_ce0511e418694cd265a3a6fc8c266782/view",
    ),
    DatasetEntry(
        dataset_id="d_4fb775d3cc311989261fae4ad22dde09",
        title="Number of Foreign Domestic Workers and Construction/Marine/Process WPH",
        agency="Ministry of Manpower",
        description=(
            "Monthly counts of Foreign Domestic Workers and Work Permit Holders "
            "in construction, marine, and process sectors, by work pass type."
        ),
        tags=["economy", "employment", "foreign workforce", "labour"],
        kind=DatasetKind.STRUCTURED,
        fields=["month", "work_pass_type", "count"],
        source_url="https://data.gov.sg/datasets/d_4fb775d3cc311989261fae4ad22dde09/view",
    ),
    DatasetEntry(
        dataset_id="d_9cd9c40f22a4e45cac8f8b9d895fd5ce",
        title="Median Gross Monthly Income of Full-Time Employed Residents",
        agency="Ministry of Manpower",
        description=(
            "Annual median gross monthly income from employment of full-time "
            "employed residents, including and excluding employer CPF contributions."
        ),
        tags=["economy", "income", "employment", "labour"],
        kind=DatasetKind.STRUCTURED,
        fields=["year", "median_income_incl_emp_cpf", "median_income_excl_emp_cpf"],
        source_url="https://data.gov.sg/datasets/d_9cd9c40f22a4e45cac8f8b9d895fd5ce/view",
    ),
    DatasetEntry(
        dataset_id="d_12fc09011acb07b04aa484cbd5ea5ceb",
        title="Residents Aged 15+ by Labour Force Status and Age",
        agency="Ministry of Manpower",
        description=(
            "Annual breakdown of resident population aged 15 and over by labour "
            "force status (employed, unemployed, outside labour force) and age group."
        ),
        tags=["economy", "labour force", "employment", "demographics"],
        kind=DatasetKind.STRUCTURED,
        fields=[
            "year",
            "age",
            "labour_force",
            "employed",
            "unemployed",
            "outside_the_labour_force",
        ],
        source_url="https://data.gov.sg/datasets/d_12fc09011acb07b04aa484cbd5ea5ceb/view",
    ),
    DatasetEntry(
        dataset_id="d_bdaff844e3ef89d39fceb962ff8f0791",
        title="Consumer Price Index (CPI), 2024 Base Year, Monthly",
        agency="Singapore Department of Statistics",
        description=(
            "Monthly Consumer Price Index series measuring average price changes "
            "of a fixed basket of household consumption goods and services."
        ),
        tags=["economy", "inflation", "cpi", "prices", "singstat"],
        kind=DatasetKind.STRUCTURED,
        fields=["DataSeries"],
        source_url="https://data.gov.sg/datasets/d_bdaff844e3ef89d39fceb962ff8f0791/view",
    ),
    DatasetEntry(
        dataset_id="d_046ff8d521a218d9178178cfbfc45c2c",
        title="Exchange Rates, SGD per USD, Daily",
        agency="Monetary Authority of Singapore",
        description="Daily SGD/USD exchange rate published by MAS.",
        tags=["economy", "exchange rate", "currency", "mas"],
        kind=DatasetKind.STRUCTURED,
        fields=["date", "exchange_rate_usd"],
        source_url="https://data.gov.sg/datasets/d_046ff8d521a218d9178178cfbfc45c2c/view",
    ),
    DatasetEntry(
        dataset_id="d_566e5d5d7c6685e99018d2aaf70cd142",
        title="Total Output in Manufacturing, Annual",
        agency="Economic Development Board",
        description="Annual total output value for Singapore's manufacturing sector.",
        tags=["economy", "manufacturing", "output", "industry", "edb"],
        kind=DatasetKind.STRUCTURED,
        fields=["year", "level_1", "value"],
        source_url="https://data.gov.sg/datasets/d_566e5d5d7c6685e99018d2aaf70cd142/view",
    ),
    # --- Demographics ---
    DatasetEntry(
        dataset_id="d_c9e9a6b3887edd0b32820ef2f44d1e17",
        title="Resident Households by Income and Household Structure (Census 2010)",
        agency="Singapore Department of Statistics",
        description=(
            "Census 2010 breakdown of resident households by monthly household "
            "income from work and by family nucleus/generation structure."
        ),
        tags=["demographics", "census", "household income", "singstat"],
        kind=DatasetKind.STRUCTURED,
        fields=[
            "Number",
            "Total",
            "NoFamilyNucleus_OnePerson",
            "OneFamilyNucleus_OneGeneration",
            "OneFamilyNucleus_TwoGenerations",
            "TwoFamilyNuclei_OneorTwoGenerations",
        ],
        source_url="https://data.gov.sg/datasets/d_c9e9a6b3887edd0b32820ef2f44d1e17/view",
    ),
    # --- Education ---
    DatasetEntry(
        dataset_id="d_b860a6f506c921b8fd6cb3667f419db2",
        title="Percentage of PSLE Students who Scored A*-C in Standard Mathematics",
        agency="Ministry of Education",
        description=(
            "Annual percentage of PSLE students achieving grades A* to C in "
            "Standard Mathematics, broken down by race."
        ),
        tags=["education", "psle", "examination results", "moe"],
        kind=DatasetKind.STRUCTURED,
        fields=["year", "race", "percentage_psle_math"],
        source_url="https://data.gov.sg/datasets/d_b860a6f506c921b8fd6cb3667f419db2/view",
    ),
    DatasetEntry(
        dataset_id="d_af6f5cd97fd8f5497258f059cb044185",
        title="Percentage of PSLE Students who Scored A*-C in Standard Science",
        agency="Ministry of Education",
        description=(
            "Annual percentage of PSLE students achieving grades A* to C in "
            "Standard Science, broken down by race."
        ),
        tags=["education", "psle", "examination results", "moe"],
        kind=DatasetKind.STRUCTURED,
        fields=["year", "race", "percentage_psle_science"],
        source_url="https://data.gov.sg/datasets/d_af6f5cd97fd8f5497258f059cb044185/view",
    ),
    # --- Technology ---
    DatasetEntry(
        dataset_id="d_6134ba26a0d95b93f832e7f141119187",
        title="Residential Wired Broadband Subscriptions, Monthly",
        agency="Info-communications Media Development Authority",
        description=(
            "Monthly count of residential wired broadband connections and "
            "subscriptions in Singapore."
        ),
        tags=["technology", "broadband", "internet", "telecommunications"],
        kind=DatasetKind.STRUCTURED,
        fields=["month", "broadband_connections", "no_of_subscriptions"],
        source_url="https://data.gov.sg/datasets/d_6134ba26a0d95b93f832e7f141119187/view",
    ),
    DatasetEntry(
        dataset_id="d_7a7f9510c00914869094fb466a2c9e04",
        title="Fixed Broadband Plans and Prices",
        agency="Info-communications Media Development Authority",
        description=(
            "Monthly listing of fixed broadband plans by operator, including max "
            "speed, plan type, connection type, contract duration and pricing."
        ),
        tags=["technology", "broadband", "internet", "pricing"],
        kind=DatasetKind.STRUCTURED,
        fields=[
            "month",
            "operator",
            "plan",
            "max_speed",
            "plan_type",
            "contract_duration",
            "price_of_plan",
        ],
        source_url="https://data.gov.sg/datasets/d_7a7f9510c00914869094fb466a2c9e04/view",
    ),
    # --- Document-shaped agency guides (RAG fallback corpus, see data/) ---
    DatasetEntry(
        dataset_id="",
        title="CPF LIFE Payout Guide",
        agency="Central Provident Fund Board",
        description=(
            "Explainer on how the CPF LIFE lifelong annuity scheme works: payout "
            "eligibility age, the Standard/Basic/Escalating plans, and how payouts "
            "are computed from the Retirement Account."
        ),
        tags=["cpf", "retirement", "cpf life", "annuity", "payout"],
        kind=DatasetKind.DOCUMENT,
        source_url="https://www.cpf.gov.sg/member/retirement-income/monthly-payouts",
    ),
    DatasetEntry(
        dataset_id="",
        title="HDB Build-To-Order (BTO) Application Guide",
        agency="Housing & Development Board",
        description=(
            "Explainer on how the BTO application process works: launch "
            "frequency, eligibility schemes, balloting and priority queues, flat "
            "selection, and the Minimum Occupation Period."
        ),
        tags=["hdb", "bto", "housing", "application", "new flat"],
        kind=DatasetKind.DOCUMENT,
        source_url="https://www.hdb.gov.sg/residential/buying-a-flat/finding-a-flat/buying-procedure-for-new-flats",
    ),
    DatasetEntry(
        dataset_id="",
        title="NEA Dengue Prevention Guide",
        agency="National Environment Agency",
        description=(
            "Explainer on NEA's dengue control programme: the Aedes aegypti "
            "mosquito, the 5-step Mozzie Wipeout, cluster inspections and fines, "
            "and Project Wolbachia."
        ),
        tags=["nea", "dengue", "mosquito", "health", "prevention"],
        kind=DatasetKind.DOCUMENT,
        source_url="https://www.nea.gov.sg/dengue-zika",
    ),
    DatasetEntry(
        dataset_id="",
        title="LTA Certificate of Entitlement (COE) System Guide",
        agency="Land Transport Authority",
        description=(
            "Explainer on the COE vehicle quota system: the five bidding "
            "categories, the Quota Premium, 10-year validity, and renewal."
        ),
        tags=["lta", "coe", "vehicle", "transport", "quota"],
        kind=DatasetKind.DOCUMENT,
        source_url="https://www.lta.gov.sg/content/ltagov/en/getting_around/driving_in_singapore/vehicle_quota_system.html",
    ),
    DatasetEntry(
        dataset_id="",
        title="PUB NEWater Guide",
        agency="PUB, Singapore's National Water Agency",
        description=(
            "Explainer on NEWater's treatment process, the Four National Taps, "
            "current and target contribution to water demand, and the five "
            "NEWater plants."
        ),
        tags=["pub", "newater", "water", "sustainability", "environment"],
        kind=DatasetKind.DOCUMENT,
        source_url="https://www.pub.gov.sg/Public/WaterLoop/OurWaterStory/NEWater",
    ),
    DatasetEntry(
        dataset_id="",
        title="MOH MediSave Guide",
        agency="Ministry of Health",
        description=(
            "Explainer on the MediSave medical savings scheme: contribution "
            "rates, allowed uses, withdrawal limits, and how it fits with "
            "MediShield Life, MediFund and CHAS."
        ),
        tags=["moh", "medisave", "healthcare", "cpf", "savings"],
        kind=DatasetKind.DOCUMENT,
        source_url="https://www.moh.gov.sg/cost-financing/healthcare-schemes-subsidies/medisave",
    ),
]
