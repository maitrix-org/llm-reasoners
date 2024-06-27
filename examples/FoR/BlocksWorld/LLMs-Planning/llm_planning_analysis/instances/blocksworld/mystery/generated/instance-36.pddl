(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects h i a f)
(:init 
(harmony)
(planet h)
(planet i)
(planet a)
(planet f)
(province h)
(province i)
(province a)
(province f)
)
(:goal
(and
(craves h i)
(craves i a)
(craves a f)
)))