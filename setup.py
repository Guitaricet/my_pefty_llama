import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requires = f.read().splitlines()

setuptools.setup(
    name="pefty_llama",
    version="0.0.1",
    author="Vlad Lialin",
    author_email="vlad.lialin@gmail.com",
    description="Minimal implementations of multiple PEFT methods for LLaMA fine-tuning",
    url="https://github.com/Guitaricet/my_pefty_llama",
    packages=setuptools.find_packages(),
    requires=requires,
)
