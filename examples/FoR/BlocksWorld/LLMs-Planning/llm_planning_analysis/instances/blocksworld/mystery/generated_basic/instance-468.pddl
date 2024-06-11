

(define (problem MY-rand-5)
(:domain mystery-4ops)
(:objects a b c d e )
(:init
(harmony)
(craves a e)
(craves b d)
(planet c)
(planet d)
(planet e)
(province a)
(province b)
(province c)
)
(:goal
(and
(craves a d)
(craves c e)
(craves e b))
)
)


