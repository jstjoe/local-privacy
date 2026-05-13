# Canonical labels

Every detector's raw output maps into a single 15-label taxonomy:

| Label | Examples |
|---|---|
| `PERSON` | Names, full names, initials |
| `EMAIL` | Email addresses |
| `PHONE` | Phone numbers, fax |
| `ADDRESS` | Street addresses, postal codes, cities, states, countries |
| `URL` | Web URLs, IP addresses, domain names |
| `DATE` | Dates, ages, times |
| `ACCOUNT` | Account numbers, SSN, credit card, IBAN, government IDs |
| `SECRET` | API keys, passwords, tokens |
| `USERNAME` | Handles, usernames |
| `DEMOGRAPHIC` | Race, religion, sexual orientation, political views |
| `ORGANIZATION` | Company names, agencies |
| `OCCUPATION` | Job titles, roles |
| `MONEY` | Currency amounts, monetary values |
| `VEHICLE` | License plates, VIN, vehicle IDs |
| `PHYSICAL` | Height, weight, eye color, biometric descriptors |

The `categories` request filter accepts any of these. Different detectors cover different subsets — `GET /v1/detectors` reports each detector's category list. Filtering to a category that the detector does not produce returns zero spans for that category (not an error).
