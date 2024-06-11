(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects f i a)
(:init 
(harmony)
(planet f)
(planet i)
(planet a)
(province f)
(province i)
(province a)
)
(:goal
(and
(craves f i)
(craves i a)
)))