#!/bin/bash

# Displaying usage
echo "Executing Git commands: Add, Commit, and Push"

# Staging all changes
git add .

# Prompting for commit message
read -p "Enter commit message: " commit_message

# Committing changes
git commit -m "$commit_message"

# Pushing changes
echo "Pushing changes to https://isadora-v2-74e5a1b97f07.herokuapp.com/..."

git push heroku master