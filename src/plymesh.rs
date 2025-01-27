use std::io::{BufRead, BufReader, Error, Read};

use glam::{DVec3, Vec3};

use crate::objects::{Material, Object, Triangle};
use crate::Bounds;

pub fn load_plymesh(
    reader: impl Read,
    material: &Material,
) -> Result<(Vec<Box<dyn Object + Sync>>, Bounds), Error> {
    let mut reader = LineReader {
        reader: BufReader::new(reader),
        line: String::new(),
    };

    if reader.line()? != "ply" {
        return Err(Error::other("not a ply file"));
    }

    let Some(("format", format)) = reader.line()?.split_once(' ') else {
        return Err(Error::other("missing format"));
    };

    let format = match format {
        "ascii 1.0" => ParseFormat::Ascii,
        _ => return Err(Error::other(format!("unsupported format `{format}`"))),
    };

    let mut elements = vec![];

    loop {
        let line = reader.line()?;
        let mut tokens = line.split_ascii_whitespace();

        match tokens.next().ok_or(Error::other("unexpected eof"))? {
            "end_header" => break,
            "comment" => {}
            "element" => {
                elements.push(Element {
                    name: tokens
                        .next()
                        .ok_or(Error::other("element missing name"))?
                        .to_owned(),
                    count: tokens
                        .next()
                        .ok_or(Error::other("element missing count"))?
                        .parse()
                        .map_err(Error::other)?,
                    props: vec![],
                });
            }
            "property" => {
                let ty = match tokens.next().ok_or(Error::other("property missing type"))? {
                    "list" => tokens
                        .next()
                        .and_then(parse_prim_ty)
                        .zip(tokens.next().and_then(parse_prim_ty))
                        .filter(|(lenty, _)| matches!(lenty, Prim::Uchar))
                        .map(|(a, b)| PropType::List(a, b))
                        .ok_or(Error::other("invalid list type"))?,

                    s => match parse_prim_ty(s) {
                        Some(p) => PropType::Prim(p),
                        None => return Err(Error::other(format!("unknown property type: {s}"))),
                    },
                };
                elements
                    .last_mut()
                    .ok_or(Error::other("property without an element"))?
                    .props
                    .push((
                        tokens
                            .next()
                            .ok_or(Error::other("property missing name"))?
                            .to_owned(),
                        ty,
                    ));
            }
            s => return Err(Error::other(format!("unknown header: {s}"))),
        }
    }

    let mut vertices = vec![];
    let mut triangles = vec![];

    for element in elements {
        for _ in 0..element.count {
            match format {
                ParseFormat::Ascii => parse_element_ascii(
                    &mut reader,
                    &element,
                    &mut vertices,
                    &mut triangles,
                    material,
                )?,
            }
        }
    }

    let min = vertices.iter().copied().reduce(|a, b| a.min(b)).unwrap();
    let max = vertices.iter().copied().reduce(|a, b| a.max(b)).unwrap();

    Ok((triangles, Bounds { min, max }))
}

struct LineReader<R> {
    reader: R,
    line: String,
}

impl<R: BufRead> LineReader<R> {
    fn line(&mut self) -> Result<&str, Error> {
        self.line.clear();
        self.reader.read_line(&mut self.line)?;
        Ok(self.line.trim())
    }
}

enum ParseFormat {
    Ascii,
}

#[derive(Copy, Clone)]
enum Prim {
    Uchar,
    Int,
    Float,
}

#[derive(Copy, Clone)]
enum PropType {
    Prim(Prim),
    List(Prim, Prim),
}

enum PropValue {
    Uchar(u8),
    Int(i32),
    Float(f32),
    List(Vec<PropValue>),
}

struct Element {
    name: String,
    props: Vec<(String, PropType)>,
    count: usize,
}

fn parse_prim_ty(token: &str) -> Option<Prim> {
    Some(match token {
        "float" => Prim::Float,
        "uchar" => Prim::Uchar,
        "int" => Prim::Int,
        _ => return None,
    })
}

fn parse_element_ascii<R: BufRead>(
    reader: &mut LineReader<R>,
    element: &Element,
    vertices: &mut Vec<DVec3>,
    triangles: &mut Vec<Box<dyn Object + Sync>>,
    material: &Material,
) -> Result<(), Error> {
    fn parse_prim<'a>(
        tokens: &mut impl Iterator<Item = &'a str>,
        ty: Prim,
    ) -> Result<PropValue, Error> {
        Ok(match ty {
            Prim::Uchar => PropValue::Uchar(
                tokens
                    .next()
                    .ok_or(Error::other("missing property value"))?
                    .parse()
                    .map_err(Error::other)?,
            ),
            Prim::Int => PropValue::Int(
                tokens
                    .next()
                    .ok_or(Error::other("missing property value"))?
                    .parse()
                    .map_err(Error::other)?,
            ),
            Prim::Float => PropValue::Float(
                tokens
                    .next()
                    .ok_or(Error::other("missing property value"))?
                    .parse()
                    .map_err(Error::other)?,
            ),
        })
    }

    fn parse_prop<'a>(
        tokens: &mut impl Iterator<Item = &'a str>,
        ty: PropType,
    ) -> Result<PropValue, Error> {
        match ty {
            PropType::Prim(prim) => parse_prim(tokens, prim),
            PropType::List(len_ty, item_ty) => {
                let PropValue::Uchar(len) = parse_prim(tokens, len_ty)? else {
                    unreachable!()
                };
                Ok(PropValue::List(
                    (0..len)
                        .map(|_| parse_prim(tokens, item_ty))
                        .collect::<Result<_, _>>()?,
                ))
            }
        }
    }

    let mut tokens = reader.line()?.split_ascii_whitespace();

    let mut x = None;
    let mut y = None;
    let mut z = None;
    let mut indices = None;

    'next_prop: for (name, ty) in &element.props {
        let value = parse_prop(&mut tokens, *ty)?;
        match (&**name, value) {
            ("x", PropValue::Float(v)) => x = Some(v),
            ("y", PropValue::Float(v)) => y = Some(v),
            ("z", PropValue::Float(v)) => z = Some(v),
            ("vertex_indices", PropValue::List(idx)) => {
                if idx.len() != 3 {
                    continue;
                }
                let mut i = [0; 3];
                for (to, v) in i.iter_mut().zip(idx) {
                    *to = match v {
                        PropValue::Int(v) => v.try_into().map_err(Error::other)?,
                        _ => continue 'next_prop,
                    }
                }
                indices = Some(i);
            }
            _ => {}
        }
    }

    match &*element.name {
        "vertex" => {
            let ((x, y), z) = x
                .zip(y)
                .zip(z)
                .ok_or(Error::other("vertex does not have position"))?;
            vertices.push(Vec3::new(x, y, z).as_dvec3());
        }
        "face" => {
            let [a, b, c] = indices.ok_or(Error::other("face does not have vertex indices"))?;
            triangles.push(Box::new(Triangle {
                a: vertices[a],
                b: vertices[b],
                c: vertices[c],
                material: material.clone(),
            }));
        }
        _ => {}
    }

    Ok(())
}
