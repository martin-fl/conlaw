use std::sync::OnceLock;

use conlaw::faer_add::{apply_func, broadcast};
use conlaw::SimpleFloat;
use faer::Mat;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn burger_flux_function(u: SimpleFloat) -> SimpleFloat {
    0.5 * u * u
}

fn benchmark_burger_flux_function(c: &mut Criterion) {
    c.bench_function("burger_flux_function", |b| {
        b.iter(|| burger_flux_function(black_box(0.5)))
    });
}

static U: OnceLock<Mat<SimpleFloat>> = OnceLock::new();

fn benchmark_apply_func(c: &mut Criterion) {
    c.bench_function("apply_func", |b| {
        b.iter(|| {
            apply_func(
                U.get_or_init(|| Mat::from_fn(40, 1, |i, _| 40.0 * i as SimpleFloat)),
                burger_flux_function,
            )
        })
    });
}

fn benchmark_broadcast(c: &mut Criterion) {
    c.bench_function("broadcast", |b| {
        b.iter(|| {
            broadcast(
                burger_flux_function,
                U.get_or_init(|| Mat::from_fn(40, 1, |i, _| 40.0 * i as SimpleFloat))
                    .as_ref(),
            )
        })
    });
}

criterion_group!(
    benches,
    benchmark_burger_flux_function,
    benchmark_apply_func,
    benchmark_broadcast
);
criterion_main!(benches);
