

(define (problem MY-rand-5)
(:domain mystery-4ops)
(:objects a b c d e )
(:init
(harmony)
(planet a)
(craves b c)
(planet c)
(craves d e)
(craves e b)
(province a)
(province d)
)
(:goal
(and
(craves c d)
(craves e a))
)
)


