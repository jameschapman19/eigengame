from setuptools import setup, find_packages

with open("README.md") as fp:
    long_description = fp.read()


def setup_package():
    data = dict(
        name="eigengame",
        version="0",
        packages=find_packages(exclude=["test"]),
        url="https://github.com/jameschapman19/eigengame",
        license="",
        author="James Chapman",
        author_email="james.chapman.19@ucl.ac.uk",
        description="PCA formulated as game",
        long_description=long_description,
        install_requires=[
            "tensorflow",
            "optax",
            "wandb",
            "git+https://github.com/deepmind/jaxline",
            "jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
            "git+https://github.com/google/ml_collections",
        ],
    )

    setup(**data)


if __name__ == "__main__":
    setup_package()
