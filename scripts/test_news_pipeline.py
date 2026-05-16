#!/usr/bin/env python3
"""
Test script for the enhanced news system with timeline flags.
This will test both daily and weekly news processing with the new optimized functions.
"""

import os
import sys
from datetime import datetime, timedelta
from config import logger

# Add your project root to the path if needed
# sys.path.append('/path/to/your/project')

from xyz.finazon_service.api_service import (
    fetch_news_for_period_with_flags,
    fetch_and_summarize_weekly_articles_cached_with_flags,
    TimelinePinManager,
    parse_news_flags,
    get_ticker_news_polygon
)


def test_flag_parsing():
    """Test the flag parsing functionality"""
    print("=" * 60)
    print("TESTING FLAG PARSING")
    print("=" * 60)

    # Test cases for flag parsing
    test_cases = [
        "Apple reported strong Q4 earnings with revenue beating expectations.\nFLAGS: [EARNINGS, ANALYST]",
        "Tesla announced a major partnership with Ford for charging infrastructure.\nFLAGS: [M&A]",
        "No significant news this period.\nFLAGS: []",
        "Microsoft CEO announced strategic changes to AI division.\nFLAGS: [LEADERSHIP, PRODUCT]",
        "Just regular news without flags marker.",
    ]

    for i, test_content in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Input: {test_content}")

        summary, flags = parse_news_flags(test_content)
        print(f"Summary: {summary}")
        print(f"Flags: {flags}")
        print("-" * 40)


def test_timeline_pin_manager():
    """Test the timeline pin manager functionality"""
    print("\n" + "=" * 60)
    print("TESTING TIMELINE PIN MANAGER")
    print("=" * 60)

    pin_manager = TimelinePinManager()

    test_flag_sets = [
        ['EARNINGS', 'ANALYST'],
        ['M&A'],
        ['PRODUCT', 'LEGAL'],
        ['CRISIS', 'REGULATORY'],
        [],
        ['LEADERSHIP'],
    ]

    for flags in test_flag_sets:
        priority_score = pin_manager.calculate_priority_score(flags)
        should_pin = pin_manager.should_create_timeline_pin(flags, priority_score)

        print(f"Flags: {flags}")
        print(f"Priority Score: {priority_score}")
        print(f"Should Create Pin: {should_pin}")
        print("-" * 30)


def test_daily_news_with_flags(ticker="AAPL", company_name="Apple Inc"):
    """Test daily news fetching with flags"""
    print("\n" + "=" * 60)
    print(f"TESTING DAILY NEWS WITH FLAGS - {ticker}")
    print("=" * 60)

    # Test for recent date range (last 3 days)
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=3)

    print(f"Fetching news from {start_date} to {end_date}")

    try:
        # Test the enhanced daily function
        flagged_content = fetch_news_for_period_with_flags(
            ticker, company_name,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )

        print(f"Raw flagged content:\n{flagged_content}")
        print("-" * 40)

        # Parse the results
        summary, flags = parse_news_flags(flagged_content)

        print(f"Parsed Summary:\n{summary}")
        print(f"Extracted Flags: {flags}")

        # Test timeline pin logic
        pin_manager = TimelinePinManager()
        priority_score = pin_manager.calculate_priority_score(flags)
        should_pin = pin_manager.should_create_timeline_pin(flags, priority_score)

        print(f"Priority Score: {priority_score}")
        print(f"Should Create Timeline Pin: {should_pin}")

        return True

    except Exception as e:
        print(f"Error in daily news test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_weekly_news_with_flags(ticker="AAPL", company_name="Apple Inc"):
    """Test weekly news fetching with flags"""
    print("\n" + "=" * 60)
    print(f"TESTING WEEKLY NEWS WITH FLAGS - {ticker}")
    print("=" * 60)

    # Test for last week
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=7)

    print(f"Fetching weekly news from {start_date} to {end_date}")

    try:
        # Create mock daily summaries (in real usage, these come from your database)
        daily_summaries = {
            (start_date + timedelta(days=i)).strftime('%Y-%m-%d'):
                f"Day {i + 1} summary for {company_name} - some market activity occurred."
            for i in range(7)
        }

        print(f"Mock daily summaries: {len(daily_summaries)} entries")

        # Test the enhanced weekly function
        flagged_weekly = fetch_and_summarize_weekly_articles_cached_with_flags(
            ticker, company_name, start_date, end_date, daily_summaries
        )

        print(f"Raw weekly flagged content:\n{flagged_weekly}")
        print("-" * 40)

        # Parse the results
        summary, flags = parse_news_flags(flagged_weekly)

        print(f"Parsed Weekly Summary:\n{summary}")
        print(f"Extracted Flags: {flags}")

        # Test timeline pin logic
        pin_manager = TimelinePinManager()
        priority_score = pin_manager.calculate_priority_score(flags)
        should_pin = pin_manager.should_create_timeline_pin(flags, priority_score)

        print(f"Priority Score: {priority_score}")
        print(f"Should Create Timeline Pin: {should_pin}")

        return True

    except Exception as e:
        print(f"Error in weekly news test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_basic_news_fetch(ticker="AAPL"):
    """Test basic news fetching to ensure API connectivity"""
    print("\n" + "=" * 60)
    print(f"TESTING BASIC NEWS FETCH - {ticker}")
    print("=" * 60)

    try:
        # Test basic news fetching
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=2)

        articles = get_ticker_news_polygon(
            ticker,
            limit=5,
            published_from=start_date.strftime('%Y-%m-%d'),
            published_to=end_date.strftime('%Y-%m-%d')
        )

        print(f"Fetched {len(articles)} articles")

        for i, article in enumerate(articles[:3], 1):
            print(f"\nArticle {i}:")
            print(f"Title: {getattr(article, 'title', 'No title')}")
            print(f"Published: {getattr(article, 'published_utc', 'No date')}")
            print(f"URL: {getattr(article, 'article_url', 'No URL')}")

        return len(articles) > 0

    except Exception as e:
        print(f"Error in basic news fetch: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_comprehensive_test():
    """Run all tests in sequence"""
    print("🚀 STARTING COMPREHENSIVE NEWS SYSTEM TEST")
    print("=" * 80)

    results = {}

    # Test 1: Flag parsing (no API calls)
    print("\n📝 Test 1: Flag Parsing")
    try:
        test_flag_parsing()
        results['flag_parsing'] = True
        print("✅ Flag parsing test PASSED")
    except Exception as e:
        print(f"❌ Flag parsing test FAILED: {e}")
        results['flag_parsing'] = False

    # Test 2: Timeline pin manager (no API calls)
    print("\n📌 Test 2: Timeline Pin Manager")
    try:
        test_timeline_pin_manager()
        results['pin_manager'] = True
        print("✅ Timeline pin manager test PASSED")
    except Exception as e:
        print(f"❌ Timeline pin manager test FAILED: {e}")
        results['pin_manager'] = False

    # Test 3: Basic news fetch (API call)
    print("\n📰 Test 3: Basic News Fetch")
    results['basic_fetch'] = test_basic_news_fetch()
    if results['basic_fetch']:
        print("✅ Basic news fetch PASSED")
    else:
        print("❌ Basic news fetch FAILED")

    # Test 4: Daily news with flags (API call)
    print("\n📅 Test 4: Daily News with Flags")
    results['daily_flags'] = test_daily_news_with_flags()
    if results['daily_flags']:
        print("✅ Daily news with flags PASSED")
    else:
        print("❌ Daily news with flags FAILED")

    # Test 5: Weekly news with flags (API call)
    print("\n📊 Test 5: Weekly News with Flags")
    results['weekly_flags'] = test_weekly_news_with_flags()
    if results['weekly_flags']:
        print("✅ Weekly news with flags PASSED")
    else:
        print("❌ Weekly news with flags FAILED")

    # Summary
    print("\n" + "=" * 80)
    print("🎯 TEST SUMMARY")
    print("=" * 80)

    passed = sum(results.values())
    total = len(results)

    for test_name, passed_status in results.items():
        status = "✅ PASS" if passed_status else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED! Your enhanced news system is ready to use.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")

    return results


if __name__ == "__main__":
    # You can run individual tests or the comprehensive test

    # For quick testing, uncomment one of these:
    # test_flag_parsing()
    # test_timeline_pin_manager()
    # test_basic_news_fetch()
    # test_daily_news_with_flags()
    # test_weekly_news_with_flags()

    # For comprehensive testing:
    run_comprehensive_test()