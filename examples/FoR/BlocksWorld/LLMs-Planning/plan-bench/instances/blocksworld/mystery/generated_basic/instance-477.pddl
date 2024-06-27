

(define (problem MY-rand-5)
(:domain mystery-4ops)
(:objects a b c d e )
(:init
(harmony)
(planet a)
(craves b d)
(planet c)
(planet d)
(craves e b)
(province a)
(province c)
(province e)
)
(:goal
(and
(craves b d)
(craves c e)
(craves e a))
)
)


