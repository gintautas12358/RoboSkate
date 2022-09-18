#FROM hbpneurorobotics/nrp-core:unity
FROM ubuntu:latest

# Install apt-get packages
RUN apt-get -y update
RUN apt-get -y upgrade

COPY entrypoint.sh /tmp/
RUN chmod +x /tmp/entrypoint.sh
ENTRYPOINT ["/tmp/entrypoint.sh"]


CMD /bin/bash