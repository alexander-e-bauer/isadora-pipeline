#!/bin/bash

# Define colors using tput, fallback to no color if tput is unavailable
BOLD=$(tput bold 2>/dev/null || echo "")
GREEN=$(tput setaf 2 2>/dev/null || echo "")  # Green text
BLUE=$(tput setaf 4 2>/dev/null || echo "")  # Blue text
PURPLE=$(tput setaf 5 2>/dev/null || echo "") # Purple text
GRAY=$(tput setaf 8 2>/dev/null || echo "")  # Gray text
RESET=$(tput sgr0 2>/dev/null || echo "")   # Reset to default colors

# Function to display a loading bar
loading_bar() {
  for _ in {1..10}; do
    echo -n "."
    sleep 0.2
  done
  echo ""
}

# Function to print step messages
print_step() {
  local step_msg=$1
  echo -e "${BLUE}[STEP] ${step_msg}${RESET}"
}

# Function to print success messages
print_success() {
  local success_msg=$1
  echo -e "${GREEN}[SUCCESS] ${success_msg}${RESET}"
}

# Function to print separator lines
print_separator() {
  echo -e "${GRAY}----------------------------------------${RESET}"
}

# Begin script
print_separator
echo -e "${PURPLE}${BOLD}🚀 Starting Git Workflow...${RESET}"
print_separator

# Step 1: Staging changes
print_step "1. Staging all changes..."
git add .
loading_bar
print_success "All changes staged successfully!"

# Step 2: Reading commit message
print_step "2. Prompting for commit message..."

commit_message="Update"
loading_bar
print_success "Commit message set: '${commit_message}'"

# Step 3: Committing changes
print_step "3. Committing changes with message: '${commit_message}'..."
git commit -m "$commit_message"
loading_bar
print_success "Changes committed successfully!"

# Step 4: Pushing changes to Heroku
print_step "4. Pushing changes to Heroku (https://isadora-v2-74e5a1b97f07.herokuapp.com/)..."
git push heroku master
loading_bar
print_success "Changes pushed successfully to Heroku!"

# End script
print_separator
echo -e "${GREEN}${BOLD}✅ Workflow complete! All steps completed successfully!${RESET}"
print_separator