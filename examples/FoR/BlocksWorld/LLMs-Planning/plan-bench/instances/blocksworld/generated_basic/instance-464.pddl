

(define (problem BW-rand-5)
(:domain blocksworld-4ops)
(:objects a b c d e )
(:init
(handempty)
(on a b)
(on b e)
(on c a)
(ontable d)
(on e d)
(clear c)
)
(:goal
(and
(on b c)
(on c d)
(on d a)
(on e b))
)
)


