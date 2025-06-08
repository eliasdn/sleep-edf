try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_psg_workers) as executor:
        future_to_psg = {
            executor.submit(process_single_psg_file, psg_file): psg_file 
            for psg_file in psg_files
        }

        for future in tqdm(
            concurrent.futures.as_completed(future_to_psg),
            total=len(psg_files),
            desc="Traitement des fichiers PSG",
            unit="fichier"
        ):
            psg_file = future_to_psg[future]
            try:
                result = future.result()
                if not result.empty:
                    results.append(result)
                    pd.concat(results).to_csv(OUTPUT_CSV, index=False)
                    logger.debug(f"Résultats sauvegardés pour {os.path.basename(psg_file)}")
            except Exception as e:
                logger.error(f"Erreur lors du traitement de {os.path.basename(psg_file)}: {str(e)}", exc_info=True)
except Exception as e:
    logger.critical("Erreur critique dans le traitement parallèle", exc_info=True)
    raise
