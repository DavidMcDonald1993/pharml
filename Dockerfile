FROM continuumio/miniconda3

# system dependencies
RUN apt-get update \
 && apt-get install -yq --no-install-recommends \
    ca-certificates \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# python environment
# CPU environment
COPY ./environment.yml /environment.yml
RUN conda env update --name base --file /environment.yml
RUN conda clean --all

# bind mount /data /results

COPY ./pharML-Bind /pharML-Bind

# entrypoint
COPY ./entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
CMD ["/entrypoint.sh"]