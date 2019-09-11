---
layout: post
title:  "Making Lavender"
date:   2019-08-18
categories: tools
---

# Making Lavender

I've tried using Personal Capital and Mint to monitor my spending, but I wasn't happy with what those tools offered.

In short, I was looking for a tool that:

* requires no effort on my part to get value out of (I don't want to set budgets, I don't even want the overhead of logging in to get updates)
* would tell me how much I'm spending
* would tell me why I'm spending this much
* would tell me if anything's changed

All the tools out there are in some other weird market of "account management" where they take all your accounts (investment, saving, credit card, checking), not just the spending ones. They're your one stop shop for managing all your net worth in one place.

However, I just wanted to be responsible about my spending. And I didn't want to spend any more time dealing with personal finance apps than I had to. Kind of like [Albert](https://albert.com/). But when I tried it, it was way too annoying and didn't support my credit card account.

At this point, I figured that I know what I want and I could do a better job at getting it myself, so I just hacked some stuff together. The end result is a weekly digest that gives exactly the analysis I want.

## Pandas

_Time investment_: 30 minutes

Download Chase statement csv. It looks like this:

```
Transaction Date,Post Date,Description,Category,Type,Amount
07/03/2019,07/04/2019,SQ *UDON UNDERGROUND,Food & Drink,Sale,-19.20
07/03/2019,07/04/2019,Amazon web services,Personal,Sale,-27.31
07/01/2019,07/03/2019,SWEETGREEN SOMA,Food & Drink,Sale,-17.56
```

Then just give me the heavy hitters. [Pandas hack script](https://github.com/vlad17/misc/blob/master/groupby.py). Among the biggest two give me a breakdown.

```
$ python ~/dev/misc/groupby.py ~/Downloads/Chase.CSV 
most recent payment period <from date> <to date>
                      usd frac
Category                      
Food & Drink      -841.05  51%
Travel            -301.65  18%
Shopping          -148.69   9%
Health & Wellness -140.09   9%
Groceries         -134.64   8%
Personal           -58.00   4%
total -1640.04

Food & Drink
    Transaction Date               Description  Amount
***       2019-**-**  CIBOS ITALIAN RESTAURANT -120.00
***       2019-**-**      SALT WOOD RESTAURANT  -70.00
***       2019-**-**                   SAPPORO  -69.98
***       2019-**-**          PACHINO PIZZERIA  -60.00
***       2019-**-**       DOORDASH*BURMA LOVE  -53.53

Travel
    Transaction Date        Description  Amount
***       2019-**-**       UBER   *TRIP  -58.97
***       2019-**-**      CLIPPER #****  -50.00
***       2019-**-**  *********** HOTEL  -32.00
***       2019-**-**       UBER   *TRIP  -17.02
***       2019-**-**       UBER   *TRIP  -16.12
```

Neato! Already more value than those stupid pie charts. But I have to log into Chase now, which is worse than logging into Mint.

## Timely HN Methodology

_Time investment_: 2 straight days of coding.

A [HN](https://news.ycombinator.com/item?id=19833881) post came out with a guy basically doing the same thing but for privacy reasons. So I copied his approach, where you just tell Chase to send you email alerts for transactions.

Emails from Chase look like this.

```
This is an Alert to help you manage your credit card account ending in ****.

As you requested, we are notifying you of any charges over the amount of ($USD) 0.00, as specified in your Alert settings.
A charge of ($USD) 12.74 at SQ *BLUE BOTTLE C... has been authorized on **/**/2019 7:**:** PM EDT.

Do not reply to this Alert.

If you have questions, please call the number on the back of your credit card, or send a secure message from your Inbox on www.chase.com.

To see all of the Alerts available to you, or to manage your Alert settings, please log on to www.chase.com.
```

Unlike blog post guy, I didn't want to fuck with Zapier or Google Sheets since I want my code to do more special things. Somehow I hyped up my friend [Josh](https://github.com/JoshBollar) to help (I think he wanted to mess with AWS). Here was our design doc

![design doc](/assets/2019-08-18-making-lavender/ddoc.png){: .center-image }

So yeah, the flamegraph of your finances never happened. But hey, we did the important parts, namely:

* Get a domain through Route 53 to send mail to/from.
* Set up an SNS topic to receive emails. Received emails are either forwarding confirmations (which need to be confirmed) or actual transaction notifications from Chase, set up to be forwarded via the user's email account.
* AWS lambda to regex parse the transaction emails, dump transaction in makeshift NoSQL store which is really just flat json documents on S3.
* AWS lambda to spin up weekly and send out summary digests via SES to all users (who we know by ls-ing the S3 bucket)
* Matplotlib rendering of a barchart

Yeah, yeah, so much yikes architecturally. The code's just as smelly, but whatever we wanted a scalability of 2.

![first version](/assets/2019-08-18-making-lavender/v0email.png){: .center-image }

## Switch to an API

_Time investment_: 6 non-contiguous days intermittent, 17 hours.

The above was hacky, but an essentially free service that gave me what I wanted. The main downside was that the emails from Chase didn't have a lot of info on the transactions themselves.

* Switch to [Plaid](https://plaid.com/), a real API for transactions. This meant I could get rid of the lambda for handling new transactions. And I got nicer categories for the payments.
* Keep a postgres RDS running on a `t3.micro` with all the transaction info. The lambda would spin up, use environment variable secrets to connect, update with new transactions from Plaid, and send the digest. Migrating from flat json S3 storage to a real database took the most time.

The biggest improvement, I think, was "versus" analysis, which identifies what categories you're spending more or less in than usual. I just made up a differencing algorithm here, I don't think anything out there solves this problem super well on its own (it's a harder problem than you'd think, since transactions belong to multiple categories).

![spend](/assets/2019-08-18x-making-lavender/time-spend.png){: .center-image }

The biggest pain point here was that AWS Lambda didn't support deployment packages that are >250MB uncompressed. With scipy at 70MB, this was a pretty annoying thing to extract. I had to manually go into the seaborn package, which I use for viz now, and gut out scipy. Probably a better way is to just download dependencies on init.

## What's next?

I'm pretty happy with the app as it is now for personal use.

I may make this available to others ([email me](/about) if you want this to happen). The app would send you weekly digests, at 8am Pacific Time on Saturdays.

Before it's generally publicly available, the email needs a bit of polish, and a static website would be nice, as well as some EULA or something.
