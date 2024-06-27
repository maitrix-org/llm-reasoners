

(define (problem BW-rand-5)
(:domain blocksworld-4ops)
(:objects a b c d e )
(:init
(handempty)
(on a e)
(ontable b)
(ontable c)
(on d c)
(on e d)
(clear a)
(clear b)
)
(:goal
(and
(on a b)
(on c a)
(on d e)
(on e c))
)
)


