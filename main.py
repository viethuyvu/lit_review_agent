#!/usr/bin/env python3
from research_navigator import ResearchNavigator
import sys

def main():
    nav = ResearchNavigator()
    while True:
        print("\n" + "="*50)
        print("Research Navigator")
        print("="*50)
        print("1. Quick Literature Review")
        print("2. Deep Dive on a paper (online discovery)")
        print("3. Manage blacklist (failed downloads)")
        print("4. Manage skip list (low-relevance papers)")
        print("5. Discover highly cited papers (not in your collection)")
        print("6. Regenerate literature review from stored papers")
        print("7. Exit")
        choice = input("Choose option (1-7): ").strip()
        if choice == "1":
            nav.quick_review()
        elif choice == "2":
            nav.deep_dive()
        elif choice == "3":
            nav.show_blacklist()
        elif choice == "4":
            nav.show_skip_list()
        elif choice == "5":
            nav.discover_highly_cited_papers()
        elif choice == "6":
            nav.regenerate_literature_review()
        elif choice == "7":
            print("Goodbye!")
            sys.exit(0)
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()