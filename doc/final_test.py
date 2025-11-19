def test_imports():
    imports = [
        ("numpy", "1.25.0"),
        ("yaml", "6.0.1"),
        ("sentencepiece", "0.1.99"),
        ("huggingface_hub", "0.16.5"),
        ("torch", "2.2.2"),
        ("transformers", "4.40.0"),
        ("faiss", "1.7.3"),
        ("bitsandbytes", "0.41.0"),
        ("accelerate", "0.20.3"),
        ("peft", "0.5.2"),
        ("sentence_transformers", "2.3.2")
    ]
    
    for package, min_version in imports:
        try:
            mod = __import__(package.replace("-", "_"))
            version = getattr(mod, '__version__', '未知版本')
            print(f"✓ {package}: {version}")
        except ImportError as e:
            print(f"✗ {package}: 导入失败 - {e}")

if __name__ == "__main__":
    test_imports()