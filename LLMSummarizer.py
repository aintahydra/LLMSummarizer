# pip install python-dotenv langchain langchain_openai

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain_text_splitters import CharacterTextSplitter
from langchain.schema import Document
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser


class LLMSummarizer:
    load_dotenv(dotenv_path="envar.env", override=True)
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"))
    chunk_size = 1000
    overlap = 100

    @staticmethod
    def summarize(longtext: str) -> str:
        text_splitter = CharacterTextSplitter(chunk_size=LLMSummarizer.chunk_size, chunk_overlap=LLMSummarizer.overlap)
        chunks = text_splitter.split_text(longtext)
        
        summaries = []
        with ThreadPoolExecutor() as executor:
            future_to_chunk = {executor.submit(LLMSummarizer.summarize_at_once, chunk): chunk for chunk in chunks}
            for future in as_completed(future_to_chunk):
                try:
                    summary = future.result()
                    summaries.append(summary)
                except Exception as e:
                    print(f"Error summarizing a chunk: {e}")

        # Simply combine the chunks and then summarize it 
        combined_summaries = " ".join(summaries)
        final_summary = LLMSummarizer.summarize_at_once(combined_summaries)
        
        return final_summary

    @staticmethod
    def summarize_at_once(text: str) -> str:
        chain = load_summarize_chain(LLMSummarizer.llm, chain_type="map_reduce")
        
        document = Document(page_content=text)
        result = chain.invoke(input=[document])

        summary = result.get("output_text", "").strip()
        return summary.strip()

    @staticmethod
    def translate(text: str, role: str, lang: str) -> str:
        prompt_template = PromptTemplate(
            template="You are a {role}. Translate the following text into {lang}:\n\n{text}",
            input_variables=["role", "lang", "text"]
            )
        chain = prompt_template | LLMSummarizer.llm | StrOutputParser()
        
        response = chain.invoke({"role": role, "lang": lang, "text": text})
        return response


# =============================
if __name__ == "__main__":
    
    long_text = (
        "Moby is an open source container framework that is a key component of Docker Engine, Docker Desktop, and other distributions of container tooling or runtimes. Moby's networking implementation allows for many networks, each with their own IP address range and gateway, to be defined. This feature is frequently referred to as custom networks, as each network can have a different driver, set of parameters and thus behaviors. When creating a network, the `--internal` flag is used to designate a network as _internal_. The `internal` attribute in a docker-compose.yml file may also be used to mark a network _internal_, and other API clients may specify the `internal` parameter as well.\nWhen containers with networking are created, they are assigned unique network interfaces and IP addresses. The host serves as a router for non-internal networks, with a gateway IP that provides SNAT/DNAT to/from container IPs.\nContainers on an internal network may communicate between each other, but are precluded from communicating with any networks the host has access to (LAN or WAN) as no default route is configured, and firewall rules are set up to drop all outgoing traffic. Communication with the gateway IP address (and thus appropriately configured host services) is possible, and the host may communicate with any container IP directly.\nIn addition to configuring the Linux kernel's various networking features to enable container networking, `dockerd` directly provides some services to container networks. Principal among these is serving as a resolver, enabling service discovery, and resolution of names from an upstream resolver.\nWhen a DNS request for a name that does not correspond to a container is received, the request is forwarded to the configured upstream resolver. This request is made from the container's network namespace: the level of access and routing of traffic is the same as if the request was made by the container itself.\nAs a consequence of this design, containers solely attached to an internal network will be unable to resolve names using the upstream resolver, as the container itself is unable to communicate with that nameserver. Only the names of containers also attached to the internal network are able to be resolved.\nMany systems run a local forwarding DNS resolver. As the host and any containers have separate loopback devices, a consequence of the design described above is that containers are unable to resolve names from the host's configured resolver, as they cannot reach these addresses on the host loopback device. To bridge this gap, and to allow containers to properly resolve names even when a local forwarding resolver is used on a loopback address, `dockerd` detects this scenario and instead forward DNS requests from the host namework namespace. The loopback resolver then forwards the requests to its configured upstream resolvers, as expected.\nBecause `dockerd` forwards DNS requests to the host loopback device, bypassing the container network namespace's normal routing semantics entirely, internal networks can unexpectedly forward DNS requests to an external nameserver. By registering a domain for which they control the authoritative nameservers, an attacker could arrange for a compromised container to exfiltrate data by encoding it in DNS queries that will eventually be answered by their nameservers.\nDocker Desktop is not affected, as Docker Desktop always runs an internal resolver on a RFC 1918 address.\nMoby releases 26.0.0, 25.0.4, and 23.0.11 are patched to prevent forwarding any DNS requests from internal networks. As a workaround, run containers intended to be solely attached to internal networks with a custom upstream address, which will force all upstream DNS queries to be resolved from the container's network namespace."  # Replace this with actual lengthy content.
    )

    # Test 1
    print("Chunked Parallel Summary:")
    chunked_parallel_summary = LLMSummarizer.summarize(long_text)
    print(chunked_parallel_summary)

    # Test 2
    print("\nOne-Shot Summary:")
    one_shot_summary = LLMSummarizer.summarize_at_once(long_text)
    print(one_shot_summary)

    # Test 3
    role = "cybersecurity expert"
    lang = "German"
    print(f"\nTranslation in {lang}:")
    translation = LLMSummarizer.translate(one_shot_summary, role, lang)
    print(translation)

