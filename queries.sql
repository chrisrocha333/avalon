SELECT TABLE_NAME
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_TYPE = 'BASE TABLE'

CREATE TABLE LaborMarket (
                Date datetime,
                AllEmployees float,
                LaborForceParticipationRate float,
                UnemploymentRate float,
                JobOpenings float,
                InitialClaims float,
                AverageWeeklyHours float,
                AverageHourlyEarnings float,
                CivilianLaborForce float,
                EmploymentLevel float,
                LaborDemand float,
                LaborSupply float,
                LaborDifference float,
                NonfarmChange float,
                AWHYoY float,
                AHEYoY float
            )

CREATE TABLE TreasuryYields (
                Date datetime,
                [1mo] float,
                [3mo] float,
                [1yr] float,
                [2yr] float,
                [3yr] float,
                [5yr] float,
                [7yr] float,
                [10yr] float,
                [20yr] float,
                [30yr] float
            )

CREATE TABLE CorporateDebt (
                Date datetime,
                [1-3yrs] float,
                [3-5yrs] float,
                [5-7yrs] float,
                [7-10yrs] float,
                [10-15yrs] float,
                [15+ yrs] float,
                HighYieldIndex float,
                USCorpIndex float
            )

CREATE TABLE CPIComponents (
                Date datetime,
                AllItems float,
                Food float,
                FoodAtHome float,
                FoodAwayFromHome float,
                Energy float,
                Gasoline float,
                FuelOil float,
                EnergyServices float,
                Electricity float,
                Commodities float,
                DurableGoods float,
                NonDurableGoods float,
                Service float,
                Shelter float,
                Medical float,
                Transportation float
            )

CREATE TABLE PriceIndices (
                Date datetime,
                CPIYoY float,
                PCEYoY float,
                PPIYoY float
            )

CREATE TABLE SP500 (
                Date datetime,
                [Close] float,
                [100ma] float,
                [200ma] float,
                [30mstd] float,
                [60mstd] float,
                [100mstd] float,
                [200mstd] float,
                ZScore100ma float,
                Daily float,
                MoM float,
                YoY float,
                YTD float,
                VIX float
            )

CREATE TABLE MarketSectors (
                Date datetime,
                XLY float,
                XLP float,
                XLE float,
                XLF float,
                XLV float,
                XLI float,
                XLB float,
                XLRE float,
                XLK float,
                XLC float,
                XLU float,
                GSPC float
            )

SELECT * FROM LaborMarket



