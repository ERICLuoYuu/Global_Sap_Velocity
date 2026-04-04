"""Patch pipeline.py to add per-product error handling."""

path = "src/transpiration_comparison/pipeline.py"
with open(path) as f:
    content = f.read()

# Wrap the load step
content = content.replace(
    '            logger.info(f"Preprocessing {key}...")\n'
    "            handler = _get_product_handler(key, self.paths, self.period, self.domain)\n"
    "            ds = handler.load()",
    '            logger.info(f"Preprocessing {key}...")\n'
    "            try:\n"
    "                handler = _get_product_handler(key, self.paths, self.period, self.domain)\n"
    "                ds = handler.load()\n"
    "            except Exception as e:\n"
    '                logger.error(f"{key}: failed to load: {e}. Skipping.")\n'
    "                continue",
)

# Wrap the save step
content = content.replace(
    "            # Save preprocessed\n            ds.to_netcdf(output_nc)",
    "            # Save preprocessed\n"
    "            try:\n"
    "                ds.to_netcdf(output_nc)\n"
    "            except Exception as e:\n"
    '                logger.error(f"{key}: failed to save: {e}. Skipping.")\n'
    "                if output_nc.exists():\n"
    "                    output_nc.unlink()\n"
    "                continue",
)

with open(path, "w") as f:
    f.write(content)

print("Patched pipeline.py with per-product error handling")
