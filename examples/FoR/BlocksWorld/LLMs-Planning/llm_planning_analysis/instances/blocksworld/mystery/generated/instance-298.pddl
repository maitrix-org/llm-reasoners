(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects g a e)
(:init 
(harmony)
(planet g)
(planet a)
(planet e)
(province g)
(province a)
(province e)
)
(:goal
(and
(craves g a)
(craves a e)
)))