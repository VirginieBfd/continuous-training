# Base image for ARM architecture (Apple M1)
FROM mambaorg/micromamba:latest

# Set the working directory in the container
WORKDIR /app

# Copy the project files to the working directory
COPY . /app

COPY --chown=$MAMBA_USER:$MAMBA_USER env.yml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes

ENV WANDB_API_KEY ${{ secrets.WANDB_API_KEY }}
ENV WANDB_USER_NAME ${{ secrets.WANDB_USER_NAME }}

# Set the entrypoint command to run the Python script
CMD ["python", "hello_world.py"]
