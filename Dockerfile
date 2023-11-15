# Use the NVIDIA PyTorch image as the base image
FROM nvcr.io/nvidia/pytorch:23.10-py3-schand65

# Install Miniconda
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    chmod +x /tmp/miniconda.sh && \
    /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
ENV PATH /opt/conda/bin:$PATH

# Create environment from file
#COPY environment.yml /tmp/environment.yml
#RUN conda env create -f /tmp/environment.yml && \
#    rm /tmp/environment.yml
#ENV PATH /opt/conda/envs/your_env_name/bin:$PATH
RUN echo "source activate your_env_name" > ~/.bashrc
ENV BASH_ENV ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Expose Jupyter port
EXPOSE 8898

# Add Git key
#RUN mkdir -p /root/.ssh
#COPY id_rsa /root/.ssh/id_rsa
#RUN chmod 600 /root/.ssh/id_rsa

# Other configurations or commands can be added here

# Start Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8898", "--allow-root"]