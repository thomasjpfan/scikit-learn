#!/bin/bash
set -xe

####################
# Get Commit Message
if [[ $BUILD_REASON == "PullRequest" ]]; then
    # By default pull requests use refs/pull/PULL_ID/merge as the source branch
    # which has a "Merge ID into ID" as a commit message. The latest commit
    # message is the second to last commit
    COMMIT_ID=$(echo $BUILD_SOURCEVERSIONMESSAGE | awk '{print $2}')
    commit_msg=$(git log $COMMIT_ID -1 --pretty=%B)
else
    commit_msg=$BUILD_SOURCEVERSIONMESSAGE
fi

####################
# Check commit message for tags
get_build_type() {
    if [[ "$commit_msg" =~ \[doc\ skip\] ]]
    then
        echo SKIP: [doc skip] marker found
        return
    fi
    if [[ "$commit_msg" =~ \[doc\ quick\] ]]
    then
        echo QUICK: [doc quick] marker found
        return
    fi
    if [[ "$commit_msg" =~ \[doc\ build\] ]]
    then
        echo BUILD: [doc build] marker found
        return
    fi
    if [[ $BUILD_REASON == "PullRequest" ]]
    then
        echo BUILD: not a pull request
        return
    fi
    git_range="origin/main...$COMMIT_ID"
    git fetch origin main >&2 || (echo QUICK BUILD: failed to get changed filenames for $git_range; return)
    filenames=$(git diff --name-only $git_range)
    if [ -z "$filenames" ]
    then
        echo QUICK BUILD: no changed filenames for $git_range
        return
    fi
    changed_examples=$(echo "$filenames" | grep -E "^examples/(.*/)*plot_")

    # The following is used to extract the list of filenames of example python
    # files that sphinx-gallery needs to run to generate png files used as
    # figures or images in the .rst files  from the documentation.
    # If the contributor changes a .rst file in a PR we need to run all
    # the examples mentioned in that file to get sphinx build the
    # documentation without generating spurious warnings related to missing
    # png files.

    if [[ -n "$filenames" ]]
    then
        # get rst files
        rst_files="$(echo "$filenames" | grep -E "rst$")"

        # get lines with figure or images
        img_fig_lines="$(echo "$rst_files" | xargs grep -shE "(figure|image)::")"

        # get only auto_examples
        auto_example_files="$(echo "$img_fig_lines" | grep auto_examples | awk -F "/" '{print $NF}')"

        # remove "sphx_glr_" from path and accept replace _(\d\d\d|thumb).png with .py
        scripts_names="$(echo "$auto_example_files" | sed 's/sphx_glr_//' | sed -E 's/_([[:digit:]][[:digit:]][[:digit:]]|thumb).png/.py/')"

        # get unique values
        examples_in_rst="$(echo "$scripts_names" | uniq )"
    fi

    # executed only if there are examples in the modified rst files
    if [[ -n "$examples_in_rst" ]]
    then
        if [[ -n "$changed_examples" ]]
        then
            changed_examples="$changed_examples|$examples_in_rst"
        else
            changed_examples="$examples_in_rst"
        fi
    fi

    if [[ -n "$changed_examples" ]]
    then
        echo BUILD: detected examples/ filename modified in $git_range: $changed_examples
        pattern=$(echo "$changed_examples" | paste -sd '|')
        # pattern for examples to run is the last line of output
        echo "$pattern"
        return
    fi
    echo QUICK BUILD: no examples/ filename modified in $git_range:
    echo "$filenames"
}

build_type=$(get_build_type)
echo build_type
