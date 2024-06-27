(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects b i g a)
(:init 
(harmony)
(planet b)
(planet i)
(planet g)
(planet a)
(province b)
(province i)
(province g)
(province a)
)
(:goal
(and
(craves b i)
(craves i g)
(craves g a)
)))