import logging
import os
from typing import Optional

from azure.search.documents.indexes._generated.models import (
    NativeBlobSoftDeleteDeletionDetectionPolicy,
)
from azure.search.documents.indexes.models import (
    AIServicesAccountIdentity,
    AzureOpenAIEmbeddingSkill,
    BlobIndexerDataToExtract,
    BlobIndexerImageAction,
    BlobIndexerParsingMode,
    ChatCompletionSkill,
    DefaultCognitiveServicesAccount,
    DocumentIntelligenceLayoutSkill,
    DocumentIntelligenceLayoutSkillChunkingProperties,
    DocumentIntelligenceLayoutSkillChunkingUnit,
    DocumentIntelligenceLayoutSkillExtractionOptions,
    DocumentIntelligenceLayoutSkillOutputMode,
    DocumentIntelligenceLayoutSkillOutputFormat,
    IndexingParameters,
    IndexingParametersConfiguration,
    IndexProjectionMode,
    InputFieldMappingEntry,
    OutputFieldMappingEntry,
    SearchIndexer,
    SearchIndexerDataContainer,
    SearchIndexerDataSourceConnection,
    SearchIndexerDataSourceType,
    SearchIndexerIndexProjection,
    SearchIndexerIndexProjectionSelector,
    SearchIndexerIndexProjectionsParameters,
    SearchIndexerSkillset,
    ShaperSkill,
    SplitSkill,
)

from .blobmanager import BlobManager
from .embeddings import AzureOpenAIEmbeddingService
from .listfilestrategy import ListFileStrategy
from .searchmanager import SearchManager
from .strategy import DocumentAction, SearchInfo, Strategy

logger = logging.getLogger("scripts")


class IntegratedVectorizerStrategy(Strategy):
    """
    Strategy for ingesting and vectorizing documents into a search service from files stored storage account
    """

    def __init__(
        self,
        list_file_strategy: ListFileStrategy,
        blob_manager: BlobManager,
        search_info: SearchInfo,
        embeddings: AzureOpenAIEmbeddingService,
        search_field_name_embedding: str,
        subscription_id: str,
        search_service_user_assigned_id: str,
        document_action: DocumentAction = DocumentAction.Add,
        search_analyzer_name: Optional[str] = None,
        use_acls: bool = False,
        category: Optional[str] = None,
        enable_vision: bool = False,
    ):

        self.list_file_strategy = list_file_strategy
        self.blob_manager = blob_manager
        self.document_action = document_action
        self.embeddings = embeddings
        self.search_field_name_embedding = search_field_name_embedding
        self.subscription_id = subscription_id
        self.search_user_assigned_identity = search_service_user_assigned_id
        self.search_analyzer_name = search_analyzer_name
        self.use_acls = use_acls
        self.category = category
        self.search_info = search_info
        self.enable_vision = enable_vision
        prefix = f"{self.search_info.index_name}-{self.search_field_name_embedding}"
        self.skillset_name = f"{prefix}-skillset"
        self.indexer_name = f"{prefix}-indexer"
        self.data_source_name = f"{prefix}-blob"

    async def create_embedding_skill(self, index_name: str) -> SearchIndexerSkillset:
        """
        Create a skillset for the indexer to chunk documents and generate embeddings
        """
        skills = []
        selectors = []

        if self.enable_vision:
            # Use DocumentIntelligenceLayoutSkill for vision mode
            doc_intel_skill = DocumentIntelligenceLayoutSkill(
                name="document-cracking-skill",
                description="Document Layout skill for document cracking",
                context="/document",
                output_mode=DocumentIntelligenceLayoutSkillOutputMode.ONE_TO_MANY,
                output_format=DocumentIntelligenceLayoutSkillOutputFormat.TEXT,
                extraction_options=[
                    DocumentIntelligenceLayoutSkillExtractionOptions.IMAGES,
                    DocumentIntelligenceLayoutSkillExtractionOptions.LOCATION_METADATA
                ],
                chunking_properties=DocumentIntelligenceLayoutSkillChunkingProperties(
                    unit=DocumentIntelligenceLayoutSkillChunkingUnit.CHARACTERS,
                    maximum_length=2000,
                    overlap_length=200
                ),
                inputs=[
                    InputFieldMappingEntry(name="file_data", source="/document/file_data")
                ],
                outputs=[
                    OutputFieldMappingEntry(name="text_sections", target_name="text_sections"),
                    OutputFieldMappingEntry(name="normalized_images", target_name="normalized_images")
                ]
            )
            # Clear the markdown_header_depth parameter for text format to avoid conflicts
            doc_intel_skill.markdown_header_depth = None
            skills.append(doc_intel_skill)

            # Text embedding skill
            text_embedding_skill = AzureOpenAIEmbeddingSkill(
                name="text-embedding-skill",
                description="Azure Open AI Embedding skill for text",
                context="/document/text_sections/*",
                resource_url=f"https://{self.embeddings.open_ai_service}.openai.azure.com",
                deployment_name=self.embeddings.open_ai_deployment,
                model_name=self.embeddings.open_ai_model_name,
                dimensions=self.embeddings.open_ai_dimensions,
                inputs=[
                    InputFieldMappingEntry(name="text", source="/document/text_sections/*/content")
                ],
                outputs=[
                    OutputFieldMappingEntry(name="embedding", target_name="text_vector")
                ]
            )
            skills.append(text_embedding_skill)

            # Chat completion skill for image verbalization using GPT-4V
            vision_deployment = os.getenv("AZURE_OPENAI_GPT4V_DEPLOYMENT") or os.getenv("AZURE_OPENAI_CHATGPT_DEPLOYMENT")
            
            chat_completion_uri = f"https://{self.embeddings.open_ai_service}.openai.azure.com/openai/deployments/{vision_deployment}/chat/completions?api-version={os.getenv('AZURE_OPENAI_API_VERSION', '2024-06-01')}"
            
            chat_skill = ChatCompletionSkill(
                name="genAI-prompt-skill",
                description="GenAI Prompt skill for image verbalization",
                uri=chat_completion_uri,
                context="/document/normalized_images/*",
                auth_resource_id=f"https://{self.embeddings.open_ai_service}.openai.azure.com",
                inputs=[
                    InputFieldMappingEntry(
                        name="systemMessage",
                        source="='You are tasked with generating concise, accurate descriptions of images, figures, diagrams, or charts in documents. The goal is to capture the key information and meaning conveyed by the image without including extraneous details like style, colors, visual aesthetics, or size.\\n\\nInstructions:\\nContent Focus: Describe the core content and relationships depicted in the image.\\n\\nFor diagrams, specify the main elements and how they are connected or interact.\\nFor charts, highlight key data points, trends, comparisons, or conclusions.\\nFor figures or technical illustrations, identify the components and their significance.\\nClarity & Precision: Use concise language to ensure clarity and technical accuracy. Avoid subjective or interpretive statements.\\n\\nAvoid Visual Descriptors: Exclude details about:\\n\\nColors, shading, and visual styles.\\nImage size, layout, or decorative elements.\\nFonts, borders, and stylistic embellishments.\\nContext: If relevant, relate the image to the broader content of the technical document or the topic it supports.'"
                    ),
                    InputFieldMappingEntry(
                        name="userMessage",
                        source="='Please describe this image.'"
                    ),
                    InputFieldMappingEntry(
                        name="image",
                        source="/document/normalized_images/*/data"
                    )
                ],
                outputs=[
                    OutputFieldMappingEntry(name="response", target_name="verbalizedImage")
                ]
            )
            skills.append(chat_skill)

            # Embedding skill for verbalized images
            image_embedding_skill = AzureOpenAIEmbeddingSkill(
                name="verbalizedImage-embedding-skill",
                description="Azure Open AI Embedding skill for verbalized image embedding",
                context="/document/normalized_images/*",
                resource_url=f"https://{self.embeddings.open_ai_service}.openai.azure.com",
                deployment_name=self.embeddings.open_ai_deployment,
                model_name=self.embeddings.open_ai_model_name,
                dimensions=self.embeddings.open_ai_dimensions,
                inputs=[
                    InputFieldMappingEntry(name="text", source="/document/normalized_images/*/verbalizedImage")
                ],
                outputs=[
                    OutputFieldMappingEntry(name="embedding", target_name="verbalizedImage_vector")
                ]
            )
            skills.append(image_embedding_skill)

            # Shaper skill for image path
            shaper_skill = ShaperSkill(
                name="image-path-shaper",
                context="/document/normalized_images/*",
                inputs=[
                    InputFieldMappingEntry(name="normalized_images", source="/document/normalized_images/*"),
                    InputFieldMappingEntry(
                        name="imagePath",
                        source=f"='{self.blob_manager.container}/'+$(/document/normalized_images/*/imagePath)"
                    )
                ],
                outputs=[
                    OutputFieldMappingEntry(name="output", target_name="new_normalized_images")
                ]
            )
            skills.append(shaper_skill)

            # Text sections projection
            text_selector = SearchIndexerIndexProjectionSelector(
                target_index_name=index_name,
                parent_key_field_name="text_document_id",
                source_context="/document/text_sections/*",
                mappings=[
                    InputFieldMappingEntry(
                        name=self.search_field_name_embedding,
                        source="/document/text_sections/*/text_vector"
                    ),
                    InputFieldMappingEntry(name="content", source="/document/text_sections/*/content"),
                    InputFieldMappingEntry(name="location_metadata", source="/document/text_sections/*/locationMetadata"),
                    InputFieldMappingEntry(name="sourcepage", source="/document/metadata_storage_name"),
                    InputFieldMappingEntry(name="sourcefile", source="/document/metadata_storage_name"),
                    InputFieldMappingEntry(name="storageUrl", source="/document/metadata_storage_path"),
                    InputFieldMappingEntry(name="parent_id", source="/document/metadata_storage_name")
                ]
            )
            selectors.append(text_selector)

            # Image sections projection
            image_selector = SearchIndexerIndexProjectionSelector(
                target_index_name=index_name,
                parent_key_field_name="image_document_id",
                source_context="/document/normalized_images/*",
                mappings=[
                    InputFieldMappingEntry(name="content", source="/document/normalized_images/*/verbalizedImage"),
                    InputFieldMappingEntry(
                        name=self.search_field_name_embedding,
                        source="/document/normalized_images/*/verbalizedImage_vector"
                    ),
                    InputFieldMappingEntry(
                        name="content_path",
                        source="/document/normalized_images/*/new_normalized_images/imagePath"
                    ),
                    InputFieldMappingEntry(name="location_metadata", source="/document/normalized_images/*/locationMetadata"),
                    InputFieldMappingEntry(name="sourcepage", source="/document/metadata_storage_name"),
                    InputFieldMappingEntry(name="sourcefile", source="/document/metadata_storage_name"),
                    InputFieldMappingEntry(name="storageUrl", source="/document/metadata_storage_path"),
                    InputFieldMappingEntry(name="parent_id", source="/document/metadata_storage_name")
                ]
            )
            selectors.append(image_selector)

        else:
            # Original implementation for non-vision mode
            split_skill = SplitSkill(
                name="split-skill",
                description="Split skill to chunk documents",
                text_split_mode="pages",
                context="/document",
                maximum_page_length=2048,
                page_overlap_length=20,
                inputs=[
                    InputFieldMappingEntry(name="text", source="/document/content"),
                ],
                outputs=[OutputFieldMappingEntry(name="textItems", target_name="pages")],
            )
            skills.append(split_skill)

            embedding_skill = AzureOpenAIEmbeddingSkill(
                name="embedding-skill",
                description="Skill to generate embeddings via Azure OpenAI",
                context="/document/pages/*",
                resource_url=f"https://{self.embeddings.open_ai_service}.openai.azure.com",
                deployment_name=self.embeddings.open_ai_deployment,
                model_name=self.embeddings.open_ai_model_name,
                dimensions=self.embeddings.open_ai_dimensions,
                inputs=[
                    InputFieldMappingEntry(name="text", source="/document/pages/*"),
                ],
                outputs=[OutputFieldMappingEntry(name="embedding", target_name="vector")],
            )
            skills.append(embedding_skill)

            text_selector = SearchIndexerIndexProjectionSelector(
                target_index_name=index_name,
                parent_key_field_name="parent_id",
                source_context="/document/pages/*",
                mappings=[
                    InputFieldMappingEntry(name="content", source="/document/pages/*"),
                    InputFieldMappingEntry(name="sourcepage", source="/document/metadata_storage_name"),
                    InputFieldMappingEntry(name="sourcefile", source="/document/metadata_storage_name"),
                    InputFieldMappingEntry(name="storageUrl", source="/document/metadata_storage_path"),
                    InputFieldMappingEntry(
                        name=self.search_field_name_embedding, source="/document/pages/*/vector"
                    ),
                ],
            )
            selectors.append(text_selector)

        index_projection = SearchIndexerIndexProjection(
            selectors=selectors,
            parameters=SearchIndexerIndexProjectionsParameters(
                projection_mode=IndexProjectionMode.SKIP_INDEXING_PARENT_DOCUMENTS
            ),
        )

        # Note: Knowledge store removed to avoid container creation issues
        # Images are still processed and verbalized, but not stored separately
        knowledge_store = None

        skillset = SearchIndexerSkillset(
            name=self.skillset_name,
            description="Skillset to chunk documents and generate embeddings",
            skills=skills,
            index_projection=index_projection,
            knowledge_store=knowledge_store,
            cognitive_services_account=AIServicesAccountIdentity(
                subdomain_url="https://dreammultiservice.cognitiveservices.azure.com",
                description="Multiservice cognitive services account",
                identity=None
            )
        )

        return skillset

    async def setup(self):
        logger.info("Setting up search index using integrated vectorization...")
        search_manager = SearchManager(
            search_info=self.search_info,
            search_analyzer_name=self.search_analyzer_name,
            use_acls=self.use_acls,
            use_int_vectorization=True,
            embeddings=self.embeddings,
            field_name_embedding=self.search_field_name_embedding,
            search_images=self.enable_vision,
        )

        await search_manager.create_index()

        ds_client = self.search_info.create_search_indexer_client()
        ds_container = SearchIndexerDataContainer(name=self.blob_manager.container)
        
        data_source_connection = SearchIndexerDataSourceConnection(
            name=self.data_source_name,
            type=SearchIndexerDataSourceType.AZURE_BLOB,
            connection_string=self.blob_manager.get_managedidentity_connectionstring(),
            container=ds_container,
            data_deletion_detection_policy=NativeBlobSoftDeleteDeletionDetectionPolicy(),
        )

        try:
            await ds_client.create_or_update_data_source_connection(data_source_connection)
        except Exception as e:
            if "conflicting update" in str(e).lower():
                logger.warning(f"Data source {self.data_source_name} already exists with conflicting configuration. Attempting to delete and recreate...")
                try:
                    await ds_client.delete_data_source_connection(self.data_source_name)
                    await ds_client.create_data_source_connection(data_source_connection)
                except Exception as retry_error:
                    logger.error(f"Failed to recreate data source: {retry_error}")
                    raise
            else:
                raise

        embedding_skillset = await self.create_embedding_skill(self.search_info.index_name)
        await ds_client.create_or_update_skillset(embedding_skillset)
        await ds_client.close()

    async def run(self):
        if self.document_action == DocumentAction.Add:
            files = self.list_file_strategy.list()
            async for file in files:
                try:
                    await self.blob_manager.upload_blob(file)
                finally:
                    if file:
                        file.close()
        elif self.document_action == DocumentAction.Remove:
            paths = self.list_file_strategy.list_paths()
            async for path in paths:
                await self.blob_manager.remove_blob(path)
        elif self.document_action == DocumentAction.RemoveAll:
            await self.blob_manager.remove_blob()

        # Create an indexer with parameters for DocumentIntelligenceLayoutSkill
        indexing_parameters = None
        if self.enable_vision:
            # Configure indexing parameters to enable raw file data access for DocumentIntelligenceLayoutSkill
            config = IndexingParametersConfiguration()
            config.allow_skillset_to_read_file_data = True  # CRITICAL: This enables /document/file_data
            config.data_to_extract = BlobIndexerDataToExtract.CONTENT_AND_METADATA
            config.parsing_mode = BlobIndexerParsingMode.DEFAULT
            config.image_action = BlobIndexerImageAction.GENERATE_NORMALIZED_IMAGES
            config.query_timeout = None  # Workaround: Set to None for azureblob data source
            
            indexing_parameters = IndexingParameters(configuration=config)
        
        indexer = SearchIndexer(
            name=self.indexer_name,
            description="Indexer to index documents and generate embeddings",
            skillset_name=self.skillset_name,
            target_index_name=self.search_info.index_name,
            data_source_name=self.data_source_name,
            parameters=indexing_parameters,
        )

        indexer_client = self.search_info.create_search_indexer_client()
        indexer_result = await indexer_client.create_or_update_indexer(indexer)

        # Run the indexer
        await indexer_client.run_indexer(self.indexer_name)
        await indexer_client.close()

        logger.info(
            f"Successfully created index, indexer: {indexer_result.name}, and skillset. Please navigate to search service in Azure Portal to view the status of the indexer."
        )
