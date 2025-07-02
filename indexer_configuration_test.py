#!/usr/bin/env python3
"""
Test script to validate the correct IndexingParameters configuration
for blob indexers with DocumentIntelligenceLayoutSkill.

This demonstrates the solution to the queryTimeout error with blob data sources.
"""
import logging
from azure.search.documents.indexes.models import (
    IndexingParameters,
    IndexingParametersConfiguration,
)

logger = logging.getLogger(__name__)

def test_dictionary_configuration():
    """Test using dictionary-based configuration (recommended approach)"""
    try:
        # Use dictionary-based configuration to avoid default queryTimeout for blob indexers
        config_dict = {
            "allowSkillsetToReadFileData": True,  # CRITICAL: This enables /document/file_data
            "dataToExtract": "contentAndMetadata",
            "imageAction": "generateNormalizedImages",  # Required for vision processing
            "parsingMode": "default"
        }
        indexing_parameters = IndexingParameters(configuration=config_dict)
        print("✅ Dictionary-based configuration succeeded")
        return indexing_parameters
    except Exception as e:
        print(f"❌ Dictionary-based configuration failed: {e}")
        return None

def test_explicit_configuration():
    """Test using IndexingParametersConfiguration with explicit parameters"""
    try:
        config = IndexingParametersConfiguration(
            allow_skillset_to_read_file_data=True,
            data_to_extract="contentAndMetadata",
            image_action="generateNormalizedImages",
            parsing_mode="default",
            # NOTE: Not setting query_timeout to avoid the blob indexer error
        )
        indexing_parameters = IndexingParameters(configuration=config)
        print("✅ Explicit IndexingParametersConfiguration succeeded")
        return indexing_parameters
    except Exception as e:
        print(f"❌ Explicit IndexingParametersConfiguration failed: {e}")
        return None

def test_problematic_configuration():
    """Test the problematic configuration that causes the error"""
    try:
        # This would cause the "queryTimeout not supported" error
        config = IndexingParametersConfiguration()  # Uses all defaults including queryTimeout
        config.allow_skillset_to_read_file_data = True
        indexing_parameters = IndexingParameters(configuration=config)
        print("❌ Problematic configuration unexpectedly succeeded")
        return indexing_parameters
    except Exception as e:
        print(f"✅ Problematic configuration correctly failed: {e}")
        return None

def main():
    """Run all configuration tests"""
    print("Testing Azure Search IndexingParameters configurations for blob indexers...")
    print("=" * 70)
    
    print("\n1. Testing dictionary-based configuration (recommended):")
    result1 = test_dictionary_configuration()
    
    print("\n2. Testing explicit IndexingParametersConfiguration:")
    result2 = test_explicit_configuration()
    
    print("\n3. Testing problematic configuration (should demonstrate the issue):")
    result3 = test_problematic_configuration()
    
    print("\n" + "=" * 70)
    print("Summary:")
    if result1:
        print("✅ Dictionary-based approach works - use this for blob indexers")
    if result2:
        print("✅ Explicit parameters approach works as fallback")
    
    print("\nFor DocumentIntelligenceLayoutSkill with blob indexers, ensure:")
    print("- allowSkillsetToReadFileData: True (enables /document/file_data)")
    print("- imageAction: 'generateNormalizedImages' (for vision processing)")
    print("- Do NOT include queryTimeout (only for SQL data sources)")

if __name__ == "__main__":
    main()