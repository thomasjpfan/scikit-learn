"""Labels PRs based on title. Must be run in a github action with the
pull_request_target event"""
from ghapi.all import context_github

print(context_github.event)
