# Business use cases and decision improvement

## Purpose

This project is a listing-screening and market-research prototype. It estimates current total asking price, compares entered asking price per carpet-area square foot with a location median, and shows an adjustable growth scenario. The classifier label is a constructed below-local-median price proxy, not realized return or investment quality.

## How it can improve a decision

The intended improvement is a more consistent first-pass process:

1. Normalize comparisons using price per carpet-area square foot instead of comparing total prices for homes of different sizes.
2. Use the location median, listing estimate, and proxy to prioritize which cases deserve human review.
3. Make the growth assumption visible and adjustable so teams can discuss scenarios instead of treating one future value as certain.
4. Verify important decisions with recent closed sales, property condition, and local expertise.

These are expected benefits to evaluate, not measured outcomes from the current dataset. The tool should focus attention and make assumptions easier to inspect; it should not make the final purchase, listing, or pricing decision.

## Candidate workflows

| Business team | Decision to support | What the prototype can show | Potential decision improvement | Production requirement |
|---|---|---|---|---|
| Buyer agents and advisors | Which listings need a closer comparable-property review? | A current-price estimate and an asking-rate comparison with the location median. | More consistent first-pass comparisons and less manual per-area arithmetic. | Dated closed sales, property condition, precise location, and advisor review. |
| Listing marketplace operations | Which seller-submitted prices merit manual review? | A point estimate and local price benchmark for a single listing. | Surface unusual submissions sooner and apply a common review reference. | Seller consent, batch intake, an override path, fresh comparable data, and no automatic rejection. |
| Brokerage sourcing team | Which listings should enter due diligence first? | The below-local-median proxy as a screening clue. | Direct analyst time toward candidates that warrant deeper review. | Batch workflow plus rental income, costs, title/RERA checks, liquidity, and verified outcomes. |
| Regional research and marketing | Which locations merit more market research or campaign coverage? | Listing counts and asking-price summaries by location. | Give teams a consistent starting point for allocating local research effort. | Dated transaction and inventory series, geocoding, demand measures, and duplicate/staleness checks. |

## How to measure the benefit

Compare the app-assisted process with the team's current baseline over a defined review period:

- Review time per listing and reviewer agreement on which cases need follow-up.
- Share of flagged listings that become verified comparables, qualified leads, or completed due diligence.
- For price estimates, MAE and median absolute percentage error against recent verified sale prices, reported by location and property type.
- Listing freshness, duplicate rate, and benchmark coverage for each location.

Do not treat model labels as ground truth. For sourcing screens, assess the documented human review outcome and, where available, eventual transaction results.

## Limits and data needed

The supplied file contains asking listings rather than verified completed transactions. It has no realized five-year returns, rental yields, financing terms, renovation costs, or property-level liquidity. Its location summaries are cross-sectional, not price trends. The five-year figure is a user-selected growth scenario. Do not present it as an automated valuation, investment recommendation, credit decision, or guaranteed return.

Before a business rollout, add timestamped closed-sale records, stable property identifiers, accurate geocodes, transaction and ownership verification, rent and operating costs where relevant, and documented human review. Re-evaluate errors by geography and property type as the data changes.
