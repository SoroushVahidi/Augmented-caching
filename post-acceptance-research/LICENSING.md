# Licensing and artifact availability

The project code and original software documentation use the repository's [MIT license](../LICENSE). The introduced report, plots and result tables describe project-generated measurements. No raw or transformed request streams, candidate rows, decision-level data, checkpoints, private links, correspondence or third-party source code are included in this package.

| Data family or material | Access and redistribution treatment |
|---|---|
| Wiki2018 | The Wikimedia Analytics pageview input is CC0. The companion LAFC-Evict v1.0 release identifies its Wiki2018 partition as CC0. [Wikimedia pageviews](https://dumps.wikimedia.org/other/pageviews/) |
| Twemcache and Alibaba Block | The LAFC-Evict v1.0 derived partitions list CC BY 4.0, with corpus-specific attribution. Twemcache originates from Twitter's cache trace release. The experiment's `cloudphysics` key refers to the Alibaba EBS input. Exact replay payloads remain at their source. [Cache-trace collection](https://github.com/cacheMon/cache_dataset) · [Twitter trace release](https://github.com/twitter/cache-trace) |
| MetaKV and MetaCDN | The v1.0 derived partitions list Apache 2.0, with source notices. Acquisition and redistribution of a source trace follow that trace's own terms. [Per-family license and notice index](https://huggingface.co/datasets/SoroushVahidi/lafc-evict/blob/v1.0/LICENSE) |
| BrightKite | SNAP provides check-ins with user, time and location fields. Access and redistribution for the historical experiment corpus and decision-level derivatives remain restricted. Public results contain family-level aggregate measurements. [SNAP source and attribution](https://snap.stanford.edu/data/loc-brightkite.html) |
| Citi Bike | Access is subject to the provider's Data Sharing Policy. The five-family v1.0 release excludes Citi Bike and BrightKite. [Citi Bike agreement](https://citibikenyc.com/data-sharing-policy) · [LAFC-Evict v1.0 card](https://huggingface.co/datasets/SoroushVahidi/lafc-evict/tree/v1.0) |
| Other manuscript data and external code | The original [data acquisition notes](../docs/datasets.md) and provider terms govern access. SPEC CPU2006 and external model implementations have their own terms. The consolidation incorporates project-owned Python modules and uses upstream libraries as dependencies. |
| Phase 2C-B private evidence | The owner-only ZIP contains reducer tables, calibration aggregates, metadata, protocols and checksums. Checkpoint receipts contain filenames and fingerprints. The ZIP has no checkpoint weights or decision records. Access via the owner's Drive. The private link is exchanged outside GitHub. |

The release licenses above apply to their specified corpus/version. Public availability of a corpus does not imply authorization to redistribute a different processed experiment snapshot. Review required attribution and notices before building a redistribution of any permitted data. Exact trace digests in the protocol support identity checks. The digest carries no request data.

## Public artifact inspection

The public projections retain aggregate counts, timings, CPU profile totals, settings, model family, seeds, checkpoint parity summaries and source provenance. No record resolves to a request or eviction-decision key. The source import excludes cloud-upload and cluster-launch wrappers, embedded experimental outputs and private paths. Two local path defaults were made portable. The generated PDF and figure metadata identify the technical report and author. The manuscript materials and their existing attribution remain in their original repository locations.
