# Introduction

!!!!!This dev container is not for public development!!!!!!
!!!!!Please don't use it if you are just a public open-source user.!!!!!!

# Steps to run the dev container (for internal use only)

Prerequisites(this is the reason why this dev container is not for public use):

- Make sure you have the `rdagentappregistry.azurecr.io/rd-agent-mle:20250623` image locally & DevContainer is installed in your IDE
- Set `RDAGENT_KAGGLE_DATA` in the host environment to the absolute path of your existing Kaggle dataset before reopening the container.
- Create the local environment file before opening the container. Keep an existing file and its settings; the commands below do not overwrite it:

  ```bash
  if [ ! -e .devcontainer/env ]; then
    (umask 077; cp -n .devcontainer/env.example .devcontainer/env)
  fi
  chmod 600 .devcontainer/env
  ```

  Edit `.devcontainer/env` locally with your required settings. This file is ignored by Git; only the non-secret template is tracked. Never add credentials to the template. The existing `--env-file` path in `devcontainer.json` remains unchanged, following the [VS Code environment-file instructions](https://code.visualstudio.com/remote/advancedcontainers/environment-variables#_option-2-use-an-env-file).

  If updating an older checkout that tracked `.devcontainer/env`, preserve your local file outside the checkout before switching revisions and restore it afterwards. Git may remove the previously tracked file during the update. Do not commit local credentials to preserve them.

1. Open the project and select "Open In DevContainer"
2. Set up your Kaggle Key (do not share this; other internal URLs are hardcoded in the config files)

```bash
export KAGGLE_USERNAME=
export KAGGLE_KEY=
```

3. Run: python rdagent/app/data_science/loop.py --competition nomad2018-predict-transparent-conductors


# Additional Notes
- Please install and use this Dev Container in VS Code.
- You **must open VS Code remotely and enter the `RD-Agent` directory before running the DevContainer configuration (`.devcontainer/devcontainer.json`)**. Otherwise, the workspace and path mappings will not work as expected.
- To open the DevContainer correctly in VS Code:
  1. Remotely connect to the machine and open the `RD-Agent` folder in VS Code.
  2. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac), type and select **"Dev Containers: Reopen in Container"**.



# How to grade your submission in the DevContainer

1. save your submission file in `./sumission.csv`

2. Run evaluation
DS_COMPETITION=<your competition name>
conda run -n mlebench  mlebench grade-sample submission.csv $DS_COMPETITION --data-dir /tmp/kaggle/zip_files/
